#!/usr/bin/env python
"""
Equilibrium Matching Policy for Robot Imitation Learning.

Strictly adheres to the original EqM formulation (arXiv 2510.02300):
1. Conservative vector field derived via scalar energy (E = dot(f, x) or -||f||^2/2).
2. Purely time-invariant architecture (no t-embedding, no positional timestep graphs).
3. Correct noise -> data velocity target scaled by the c(gamma) schedule.
"""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

from .configuration_eqm import EqMConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level policy wrapper
# ─────────────────────────────────────────────────────────────────────────────

class EqMPolicy(PreTrainedPolicy):
    """Equilibrium Matching Policy for Visuomotor Policy Learning."""

    config_class = EqMConfig
    name = "eqm"

    def __init__(self, config: EqMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.eqm_model = EquilibriumModel(config)
        self.last_is_ood = False
        self.reset()

    def get_optim_params(self) -> dict:
        return self.eqm_model.parameters()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION:    deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        self.last_is_ood = False

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions, is_ood = self.eqm_model.generate_actions(batch, noise=noise)
        self.last_is_ood = is_ood.any().item()
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        loss = self.eqm_model.compute_loss(batch)
        return loss, None


# ─────────────────────────────────────────────────────────────────────────────
#  Core EqM model
# ─────────────────────────────────────────────────────────────────────────────

class EquilibriumModel(nn.Module):
    def __init__(self, config: EqMConfig):
        super().__init__()
        self.config = config

        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        cond_dim_total = global_cond_dim * config.n_obs_steps

        if self.config.model_type == "unet":
            self.network = ConditionalUnet1d(config, global_cond_dim=cond_dim_total)
        elif self.config.model_type == "transformer":
            self.network = EqMTransformer1d(config, global_cond_dim=cond_dim_total)
        elif self.config.model_type == "cnn":
            raise NotImplementedError("CNN architecture not yet integrated.")
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

        if config.compile_model:
            self.network = torch.compile(self.network, mode=config.compile_mode)

        self.ebm = getattr(config, "ebm_variant", "dot")

    # ── c(γ) schedule ────────────────────────────────────────────────────────
    
    def get_c_lambda(self, lam: Tensor) -> Tensor:
        # Graceful fallback to linear if eqm_schedule_type is not in config
        schedule = getattr(self.config, "eqm_schedule_type", "linear")
        if schedule == "linear":
            return 1.0 - lam
        elif schedule == "softmax":
            return (torch.exp(-lam) - math.exp(-1.0)) / (1.0 - math.exp(-1.0))
        elif schedule == "piecewise":
            return torch.where(lam < 0.5, torch.ones_like(lam), 2.0 * (1.0 - lam))
        elif schedule == "grad_multiplier":
            return (1.0 - lam) ** 2
        return 1.0 - lam

    # ── Conservative gradient field ───────────────────────────────────────────

    def _network_output(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Raw, time-invariant network output f(x)."""
        return self.network(x, global_cond=global_cond)

    def _scalar_energy(self, x: Tensor, f: Tensor) -> Tensor:
        if self.ebm == "dot":
            return torch.sum(f * x, dim=(1, 2))
        elif self.ebm == "l2":
            return -0.5 * torch.sum(f * f, dim=(1, 2))
        else:
            raise ValueError(f"Unknown ebm variant: {self.ebm}")

    def get_gradient_field(
        self, x: Tensor, global_cond: Tensor | None = None
    ) -> Tensor:
        if self.ebm == "none":
            return self._network_output(x, global_cond)

        x_in = x.detach().requires_grad_(True)
        f = self._network_output(x_in, global_cond)
        E = self._scalar_energy(x_in, f)
        grad = torch.autograd.grad(
            outputs=E.sum(),
            inputs=x_in,
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True,
        )[0]
        return grad

    def forward_score(
        self, x_in: Tensor, global_cond: Tensor | None = None
    ) -> Tensor:
        with torch.inference_mode(False), torch.enable_grad():
            x = x_in.clone()
            gc = global_cond.clone() if global_cond is not None else None
            score = self.get_gradient_field(x, global_cond=gc)
            return score.detach()

    # ── Reverse process: noise → action ──────────────────────────────────────

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        device = get_device_from_parameters(self)
        dtype  = get_dtype_from_parameters(self)

        sample = (
            noise if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype, device=device, generator=generator,
            )
        )

        lr       = getattr(self.config, "eqm_lr", 1e-3)
        momentum = getattr(self.config, "eqm_momentum", 0.9)
        velocity = torch.zeros_like(sample)
        N        = getattr(self.config, "eqm_inference_steps", 50)
        ood_thr  = getattr(self.config, "ood_threshold", 10.0)

        for _ in range(N):
            grad = self.forward_score(sample, global_cond=global_cond)

            if self.config.eqm_sampler_type == "gd":
                sample = sample - lr * grad
            elif self.config.eqm_sampler_type == "nag_gd":
                velocity = momentum * velocity + grad
                sample   = sample - lr * velocity
            elif self.config.eqm_sampler_type == "ode":
                step_size = 1.0 / N
                sample    = sample - step_size * grad
            elif self.config.eqm_sampler_type == "adaptive":
                sample = sample - lr * grad
                grad_norms = torch.norm(grad.reshape(batch_size, -1), dim=1)
                if grad_norms.max() < ood_thr:
                    break

        clip_range = getattr(self.config, "clip_sample_range", None)
        if getattr(self.config, "clip_sample", False) and clip_range is not None:
            sample = torch.clamp(sample, -clip_range, clip_range)

        final_grad = self.forward_score(sample, global_cond=global_cond)
        grad_norms = torch.norm(final_grad.reshape(batch_size, -1), dim=1)
        is_ood     = grad_norms > ood_thr

        return sample, is_ood

    # ── Observation conditioning ──────────────────────────────────────────────

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(
                    batch[OBS_IMAGES], "b s n ... -> n (b s) ..."
                )
                img_features_list = torch.cat([
                    encoder(images)
                    for encoder, images in zip(
                        self.rgb_encoder, images_per_camera, strict=True
                    )
                ])
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps,
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond = self._prepare_global_conditioning(batch)
        actions, is_ood = self.conditional_sample(
            batch_size, global_cond=global_cond, noise=noise
        )
        start   = n_obs_steps - 1
        end     = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions, is_ood

    # ── Training loss ─────────────────────────────────────────────────────────

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        global_cond = self._prepare_global_conditioning(batch)

        x_0    = batch[ACTION]          
        B      = x_0.shape[0]
        device = x_0.device

        lam = torch.rand(B, device=device).clamp(min=1e-4)
        lam_ = lam.view(B, 1, 1)

        eps   = torch.randn_like(x_0)

        x_lam = (1.0 - lam_) * x_0 + lam_ * eps

        # Correct Target: Velocity direction is Noise -> Data, scaled by c(gamma)
        c_lam = self.get_c_lambda(lam).view(B, 1, 1)
        target = (x_0 - eps) * c_lam                     

        pred = self.get_gradient_field(x_lam, global_cond=global_cond)

        loss = F.mse_loss(pred, target, reduction="none")

        if getattr(self.config, "do_mask_loss_for_padding", False):
            in_episode_bound = ~batch["action_is_pad"]
            mask      = in_episode_bound.unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Vision encoder utilities
# ─────────────────────────────────────────────────────────────────────────────

class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features  = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.view(-1, self._out_c, 2)


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if parents:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


class DiffusionRgbEncoder(nn.Module):
    def __init__(self, config: EqMConfig):
        super().__init__()
        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        crop_shape = getattr(config, "crop_shape", None)
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(crop_shape)
                if getattr(config, "crop_is_random", False)
                else self.center_crop
            )
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )

        images_shape = next(iter(config.image_features.values())).shape
        if crop_shape is not None:
            dummy_shape_h_w = crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape   = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        num_kp = getattr(config, "spatial_softmax_num_keypoints", 32)
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=num_kp)
        self.feature_dim = num_kp * 2
        self.out  = nn.Linear(num_kp * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.relu(self.out(x))


# ─────────────────────────────────────────────────────────────────────────────
#  UNet Backbone (Strictly Time-Invariant)
# ─────────────────────────────────────────────────────────────────────────────

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias  = cond_embed[:, self.out_channels :]
            out   = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class ConditionalUnet1d(nn.Module):
    def __init__(self, config: EqMConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        # Ensure cond_dim is > 0 for nn.Linear to initialize correctly
        cond_dim = max(global_cond_dim, 1)
        
        in_out   = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )
        common = {
            "cond_dim":                  cond_dim,
            "kernel_size":               getattr(config, "kernel_size", 5),
            "n_groups":                  getattr(config, "n_groups", 8),
            "use_film_scale_modulation": getattr(config, "use_film_scale_modulation", True),
        }

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1d(dim_in, dim_out, **common),
                ConditionalResidualBlock1d(dim_out, dim_out, **common),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
            ]))

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common),
            ConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common),
        ])

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1d(dim_in * 2, dim_out, **common),
                ConditionalResidualBlock1d(dim_out, dim_out, **common),
                nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=common["kernel_size"]),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")

        # Handle purely unconditional edge cases gracefully
        if global_cond is None:
            global_feature = torch.zeros(x.shape[0], 1, device=x.device)
        else:
            global_feature = global_cond

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b d t -> b t d")


# ─────────────────────────────────────────────────────────────────────────────
#  Transformer Backbone (Strictly Time-Invariant)
# ─────────────────────────────────────────────────────────────────────────────

def _get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega  = 1.0 / (10000 ** omega)           
    pos    = np.arange(length, dtype=np.float64)
    out    = np.outer(pos, omega)              
    emb    = np.concatenate([np.sin(out), np.cos(out)], axis=1)  
    return emb.astype(np.float32)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class EqMTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x_norm = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class EqMTransformerFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class EqMTransformer1d(nn.Module):
    def __init__(self, config: EqMConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        action_dim  = config.action_feature.shape[0]
        seq_len     = config.horizon

        hidden_size = getattr(config, "transformer_hidden_size", 512)
        depth       = getattr(config, "transformer_depth",       6)
        num_heads   = getattr(config, "transformer_num_heads",   8)
        mlp_ratio   = getattr(config, "transformer_mlp_ratio",   4.0)

        self.token_embed = nn.Linear(action_dim, hidden_size, bias=True)

        pos_emb = _get_1d_sincos_pos_embed(hidden_size, seq_len)      
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos_emb).unsqueeze(0)       
        )

        # Global conditioning projection (acts as sole modulation source)
        cond_dim = max(global_cond_dim, 1)
        self.cond_proj = nn.Linear(cond_dim, hidden_size, bias=True)

        self.blocks = nn.ModuleList([
            EqMTransformerBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.final_layer = EqMTransformerFinalLayer(hidden_size, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

    def forward(
        self, x: Tensor, global_cond: Tensor | None = None
    ) -> Tensor:
        x = self.token_embed(x) + self.pos_embed   
        
        if global_cond is not None:
            c = self.cond_proj(global_cond) 
        else:
            c = torch.zeros(x.shape[0], self.cond_proj.out_features, device=x.device) 

        for block in self.blocks:
            x = block(x, c)                         

        x = self.final_layer(x, c)                  
        return x