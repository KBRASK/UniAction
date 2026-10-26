from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import tomllib

from .profiles import get_wan_profile


@dataclass(frozen=True)
class ModelConfig:
    model_root: Path
    common_root: Path
    tokenizer_root: Path
    family: str = "Wan2.2-I2V-A14B-HighNoise"
    model_id: str = "Wan-AI/Wan2.2-I2V-A14B"
    dit_pattern: str = "high_noise_model/diffusion_pytorch_model-*.safetensors"
    vae_file: str = "Wan2.1_VAE.pth"
    t5_file: str = "models_t5_umt5-xxl-enc-bf16.pth"
    dtype: str = "bfloat16"
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16
    dit2_pattern: str | None = None
    image_encoder_file: str | None = None
    switch_dit_boundary: float = 0.875
    dit_bf16_pattern: str | None = None
    dit_bf16_manifest: Path | None = None

    @property
    def profile(self):
        return get_wan_profile(self.family)


@dataclass(frozen=True)
class ActionConfig:
    adapter_l_max: int = 32
    hand_vocab_size: int = 16
    camera_vocab_size: int = 16
    robot_vocab_size: int = 12
    force_start_token_id: int = 0
    force_endpoint_token_id: int = 3
    reference_count_init: float = 784.0
    time_group_size: int = 4


@dataclass(frozen=True)
class OptimizationConfig:
    lora_rank: int = 128
    lora_alpha: int = 128


@dataclass(frozen=True)
class EvaluationConfig:
    solver: str = "euler"
    steps: int = 20
    cfg_scale: float = 1.0
    sigma_shift: float = 5.0
    seed: int = 1234


@dataclass(frozen=True)
class ArtifactConfig:
    stage1_checkpoint: Path
    stage2_checkpoint: Path
    stage2_n_ref: float = 784.0
    trained: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    artifacts: ArtifactConfig
    action: ActionConfig = field(default_factory=ActionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        model = self.model
        if model.family != "Wan2.2-I2V-A14B-HighNoise" or model.dit2_pattern is not None:
            raise ValueError("UniAction uses the Wan2.2 A14B high-noise expert at every timestep")
        if (model.height, model.width, model.num_frames, model.fps) != (480, 832, 81, 16):
            raise ValueError("The released model uses 81 frames at 832x480 and 16 fps")
        if self.action != ActionConfig() or self.optimization != OptimizationConfig():
            raise ValueError(
                "Action architecture and LoRA configuration must match the released weights"
            )
        sampling = self.evaluation
        if (
            isinstance(sampling.steps, bool)
            or not isinstance(sampling.steps, int)
            or sampling.steps < 1
        ):
            raise ValueError("steps must be a positive integer")
        if sampling.solver not in ("euler", "unipc"):
            raise ValueError("solver must be euler or unipc")
        for name in ("cfg_scale", "sigma_shift"):
            value = getattr(sampling, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def make_config(
    base_model: str | Path, checkpoint_dir: str | Path, *, trained=False, **sampling
) -> ExperimentConfig:
    base = Path(base_model).expanduser().resolve()
    weights = Path(checkpoint_dir).expanduser().resolve()
    artifacts = ArtifactConfig(weights / "stage1.safetensors", weights / "stage2.safetensors")
    if trained:
        from .checkpoints import read_release

        read_release(weights, trained=True)
        artifacts = ArtifactConfig(
            weights / "stage1.safetensors", weights / "stage2.safetensors", trained=True
        )
    config = ExperimentConfig(
        model=ModelConfig(
            model_root=base, common_root=base, tokenizer_root=base / "google/umt5-xxl"
        ),
        artifacts=artifacts,
        evaluation=EvaluationConfig(**sampling),
    )
    config.validate()
    return config


def load_config(path: str | Path, *, repo_root: str | Path | None = None) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) - {"paths", "sampling"}:
        raise ValueError("Configuration accepts only [paths] and [sampling]")
    root = Path(repo_root).resolve() if repo_root is not None else path.parent
    paths = raw["paths"]

    def resolve(value):
        value = Path(value).expanduser()
        return value if value.is_absolute() else root / value

    return make_config(
        resolve(paths["base_model"]), resolve(paths["checkpoint_dir"]), **raw.get("sampling", {})
    )
