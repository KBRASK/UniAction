from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ConditioningMode = Literal["fused_first_frame", "i2v_y"]


@dataclass(frozen=True)
class WanModelProfile:
    family: str
    model_id: str
    vae_file: str
    t5_file: str
    image_encoder_file: str | None
    conditioning: ConditioningMode
    latent_channels: int
    vae_spatial_factor: int
    vae_condition_channels: int
    requires_clip: bool
    dual_dit: bool
    hidden_dim: int
    num_blocks: int
    num_heads: int
    in_dim: int
    out_dim: int
    allowed_num_frames: tuple[int, ...]

    @property
    def expert_names(self) -> tuple[str, ...]:
        return ("dit", "dit2") if self.dual_dit else ("dit",)

    def expected_checkpoint_tensors(self, stage: int) -> int:
        experts = len(self.expert_names)
        if stage == 1:
            return experts * self.num_blocks * 10 * 2
        if stage == 2:
            return experts * self.num_blocks * 6 * 2 + 10
        raise ValueError(f"stage must be 1 or 2, got {stage}")


WAN_MODEL_PROFILES = {
    "Wan2.2-I2V-A14B-HighNoise": WanModelProfile(
        family="Wan2.2-I2V-A14B-HighNoise",
        model_id="Wan-AI/Wan2.2-I2V-A14B",
        vae_file="Wan2.1_VAE.pth",
        t5_file="models_t5_umt5-xxl-enc-bf16.pth",
        image_encoder_file=None,
        conditioning="i2v_y",
        latent_channels=16,
        vae_spatial_factor=8,
        vae_condition_channels=20,
        requires_clip=False,
        dual_dit=False,
        hidden_dim=5120,
        num_blocks=40,
        num_heads=40,
        in_dim=36,
        out_dim=16,
        allowed_num_frames=(81,),
    ),
}


def get_wan_profile(family: str) -> WanModelProfile:
    try:
        return WAN_MODEL_PROFILES[family]
    except KeyError as error:
        supported = ", ".join(sorted(WAN_MODEL_PROFILES))
        raise ValueError(f"unsupported Wan family {family!r}; choose from {supported}") from error
