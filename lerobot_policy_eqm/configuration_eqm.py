#!/usr/bin/env python

from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig

@PreTrainedConfig.register_subclass("eqm")
@dataclass
class EqMConfig(PreTrainedConfig):
    """Configuration class for Equilibrium Matching Policy."""

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 7  

    # Architecture / modeling.
    vision_backbone: str = "resnet18" # Configurable: can be swapped for jepa/dinov2 in processor
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    
    # Predictor Model Type
    model_type: str = "unet"  # Options: "unet", "transformer", "cnn"
    
    # Unet Specifics
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # EqM Specific configurations
    eqm_schedule_type: str = "linear"  # Options: "linear", "softmax", "piecewise", "grad_multiplier"
    eqm_sampler_type: str = "gd"       # Options: "gd", "nag_gd", "ode", "adaptive"
    eqm_train_timesteps: int = 100     # Granularity of lambda discretization for training
    eqm_inference_steps: int = 50      # Number of steps for solvers
    eqm_lr: float = 0.1                # Step size for GD / NAG-GD
    eqm_momentum: float = 0.9          # Momentum for NAG-GD
    ood_threshold: float = 0.05        # Threshold for gradient magnitude to flag OOD
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Optimization & System
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        supported_samplers = ["gd", "nag_gd", "ode", "adaptive"]
        if self.eqm_sampler_type not in supported_samplers:
            raise ValueError(f"`eqm_sampler_type` must be one of {supported_samplers}.")
            
        supported_schedules = ["linear", "softmax", "piecewise", "grad_multiplier"]
        if self.eqm_schedule_type not in supported_schedules:
            raise ValueError(f"`eqm_schedule_type` must be one of {supported_schedules}.")

        supported_models = ["unet", "transformer", "cnn"]
        if self.model_type not in supported_models:
            raise ValueError(f"`model_type` must be one of {supported_models}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("Provide at least one image or environment state.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None