#!/usr/bin/env python
"""
Equilibrium Matching Policy for Robot Imitation Learning.

Corrected to match the original EqM paper (arXiv 2510.02300) exactly by
enforcing a conservative gradient field derived from a scalar energy function.

KEY DIVERGENCES FIXED
─────────────────────────────────────────────────────────────────────────────
Bug 1 – Non-Conservative Vector Field (Erratic Inference Dynamics)
  The prior policy directly predicted a vector field f(x). Without a scalar 
  energy constraint, the field is non-conservative, inducing limit cycles 
  and unstable inference loops. We now compute a scalar energy E(x) and 
  derive f(x) = ∇_x E(x) via autograd.

Bug 2 – Missing Autograd Graph Tracking during Loss Computation
  When training an EBM via matching target vector fields, backpropagation must 
  flow through the first derivative. We now use torch.autograd.grad with 
  create_graph=True during compute_loss.
"""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

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
#  Top-level policy wrapper  (public API — unchanged)
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

        if self.config.model_type == "unet":
            self.network = ConditionalUnet1d(
                config, global_cond_dim=global_cond_dim * config.n_obs_steps
            )
        elif self.config.model_type == "transformer":
            raise NotImplementedError("Transformer architecture not yet integrated.")
        elif self.config.model_type == "cnn":
            raise NotImplementedError("CNN architecture not yet integrated.")

        if config.compile_model:
            self.network = torch.compile(self.network, mode=config.compile_mode)

    # ── c(γ) schedule ────────────────────────────────────────────────────────

    def get_c_lambda(self, lam: Tensor) -> Tensor:
        if self.config.eqm_schedule_type == "linear":
            return 1.0 - lam
        elif self.config.eqm_schedule_type == "softmax":
            return (torch.exp(-lam) - math.exp(-1.0)) / (1.0 - math.exp(-1.0))
        elif self.config.eqm_schedule_type == "piecewise":
            return torch.where(lam < 0.5, torch.ones_like(lam), 2.0 * (1.0 - lam))
        elif self.config.eqm_schedule_type == "grad_multiplier":
            return (1.0 - lam) ** 2
        return 1.0 - lam

    # ── Conservative Gradient Field Routing via Autograd ─────────────────────

    def get_gradient_field(self, x: Tensor, t: Tensor, global_cond: Tensor | None = None) -> Tensor:
        """
        Computes the conservative gradient field f(x) = ∇_x E(x) via autograd.
        This forces the network output trajectory to remain stable.
        """
        # Ensure input leaf tracks gradient graph modifications
        x_grad = x.detach().requires_grad_(True)
        
        # Scalar energy evaluation
        energy = self.network.forward_energy(x_grad, t, global_cond=global_cond)
        
        # Compute gradient field vector w.r.t our input manifold position
        grad = torch.autograd.grad(
            outputs=energy,
            inputs=x_grad,
            grad_outputs=torch.ones_like(energy),
            create_graph=self.training,  # Critical: True during training for weight updates
            retain_graph=self.training,
            only_inputs=True
        )[0]
        
        return grad

    # def forward_score(
    #     self,
    #     x_in: Tensor,
    #     global_cond: Tensor | None = None,
    # ) -> Tensor:
    #     """Evaluate the learned equilibrium gradient field at inference step."""
    #     device = x_in.device
    #     B = x_in.shape[0]
    #     t_zero = torch.zeros(B, dtype=torch.long, device=device)

    #     # Force gradient mapping generation during outer evaluation harness no_grad locks
    #     with torch.enable_grad():
    #         score = self.get_gradient_field(x_in, t_zero, global_cond=global_cond)

    #     return score.detach()
    def forward_score(
        self,
        x_in: Tensor,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the learned equilibrium gradient field at inference step."""
        # 1. Temporarily suspend the outer inference_mode and turn on autograd graph tracking
        with torch.inference_mode(False), torch.enable_grad():
            device = x_in.device
            B = x_in.shape[0]

            # 2. Convert incoming "Inference Tensors" to normal tensors by cloning them.
            #    This creates a valid leaf node that the autograd engine can track.
            x_in_normal = x_in.clone()
            global_cond_normal = global_cond.clone() if global_cond is not None else None
            t_zero = torch.zeros(B, dtype=torch.long, device=device)

            # 3. Compute the conservative gradient field vector
            score = self.get_gradient_field(x_in_normal, t_zero, global_cond=global_cond_normal)

            # 4. Detach the result so it safely returns back to LeRobot's inference stream
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

        lr       = self.config.eqm_lr
        momentum = self.config.eqm_momentum
        velocity = torch.zeros_like(sample)
        N        = self.config.eqm_inference_steps

        for step in range(N):
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
                if grad_norms.max() < self.config.ood_threshold:
                    break

        if self.config.clip_sample:
            sample = torch.clamp(
                sample, -self.config.clip_sample_range, self.config.clip_sample_range
            )

        final_grad = self.forward_score(sample, global_cond=global_cond)
        grad_norms = torch.norm(final_grad.reshape(batch_size, -1), dim=1)
        is_ood     = grad_norms > self.config.ood_threshold

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

        x_0 = batch[ACTION]
        B   = x_0.shape[0]
        device = x_0.device

        lam = torch.rand(B, device=device).clamp(min=1e-4)

        eps   = torch.randn_like(x_0)
        lam_  = lam.view(B, 1, 1)
        x_lam = lam_ * x_0 + (1.0 - lam_) * eps

        c_lam  = self.get_c_lambda(lam).view(B, 1, 1)
        target = (eps - x_0) * c_lam

        # Generate predictions through our autograd wrapper
        t_zero = torch.zeros(B, dtype=torch.long, device=device)
        pred   = self.get_gradient_field(x_lam, t_zero, global_cond=global_cond)

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            in_episode_bound = ~batch["action_is_pad"]
            mask     = in_episode_bound.unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Vision encoder utilities  (structurally identical)
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

        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(crop_shape)
                if config.crop_is_random
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
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape   = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out  = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.relu(self.out(x))


# ─────────────────────────────────────────────────────────────────────────────
#  UNet Backbone (Equipped with Energy Mapping Functions)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device   = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


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

        self.step_encoder = nn.Sequential(
            PositionalEmbed(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        in_out   = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )
        common = {
            "cond_dim":                  cond_dim,
            "kernel_size":               config.kernel_size,
            "n_groups":                  config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
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
            Conv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

        # Explicit Scalar Energy output configuration head
        # Safe fallback check in case config class doesn't define the parameter explicitly yet
        self.ebm_variant = getattr(config, "ebm_variant", "residual")
        if self.ebm_variant == "scalar":
            self.energy_head = nn.Sequential(
                Conv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(config.down_dims[0], 1)
            )

    def forward_energy(self, x: Tensor, lam_time: Tensor, global_cond=None) -> Tensor:
        """Computes the scalar Energy scalar E(x) across the action sequence batch."""
        x = einops.rearrange(x, "b t d -> b d t")

        lam_embed      = self.step_encoder(lam_time)
        global_feature = (
            torch.cat([lam_embed, global_cond], axis=-1)
            if global_cond is not None
            else lam_embed
        )

        encoder_skip_features = []
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

        if self.ebm_variant == "scalar":
            # Variant A: Direct global pool scalar projection head
            return self.energy_head(x)
        else:
            # Variant B: Square residual velocity norm definition: E(x) = 0.5 * ||UNet(x)||^2
            x = self.final_conv(x)
            x = einops.rearrange(x, "b d t -> b t d")
            return 0.5 * torch.sum(x ** 2, dim=(1, 2), keepdim=True)

    def forward(self, x: Tensor, lam_time: Tensor, global_cond=None) -> Tensor:
        """Legacy placeholder for standard vector field inference backward-compatibility."""
        x = einops.rearrange(x, "b t d -> b d t")
        lam_embed      = self.step_encoder(lam_time)
        global_feature = (
            torch.cat([lam_embed, global_cond], axis=-1)
            if global_cond is not None
            else lam_embed
        )
        encoder_skip_features = []
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