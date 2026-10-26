from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from uniaction.config import ActionConfig, ModelConfig


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 7.5e-5
    weight_decay: float = 0.01
    lora_rank: int = 128
    lora_alpha: int = 128
    lora_targets: tuple[str, ...] = ("q", "k", "v", "o", "ffn.0", "ffn.2")
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    warmup_updates: int = 100


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    seed: int
    stage1: SimpleNamespace
    stage2: SimpleNamespace
    artifacts: SimpleNamespace
    action: ActionConfig = field(default_factory=ActionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


def make_training_config(
    base_model, stage, *, empty_context=None, stage1_checkpoint=None, bf16_manifest=None
):
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    base = Path(base_model).expanduser().resolve()
    checkpoint = Path(stage1_checkpoint).resolve() if stage1_checkpoint else None
    if stage == 2 and (checkpoint is None or not checkpoint.is_file()):
        raise ValueError("Stage 2 requires the completed Stage 1 checkpoint")
    model_args = {}
    if bf16_manifest:
        manifest = Path(bf16_manifest).resolve()
        model_args = dict(
            dit_bf16_pattern=str(manifest.parent.relative_to(base) / "*.safetensors"),
            dit_bf16_manifest=manifest,
        )
    return TrainingConfig(
        model=ModelConfig(
            model_root=base, common_root=base, tokenizer_root=base / "google/umt5-xxl", **model_args
        ),
        seed=1 if stage == 1 else 101,
        stage1=SimpleNamespace(
            prompt_dropout=0.1,
            empty_prompt_context=Path(empty_context).resolve() if empty_context else None,
        ),
        stage2=SimpleNamespace(
            prompt_dropout=0.0, stage1_checkpoint=checkpoint, empty_prompt_context=None
        ),
        artifacts=SimpleNamespace(stage1_checkpoint=checkpoint),
    )
