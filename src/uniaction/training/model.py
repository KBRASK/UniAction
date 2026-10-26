from __future__ import annotations
import os
import random
import time
from pathlib import Path
from typing import Any
import torch
from torch import nn
from accelerate.utils import set_seed
from safetensors.torch import load_file
from uniaction.backbones.wan.loader import LocalWeightBroadcast
from uniaction.model.action import (
    ActionAdapter,
    build_action_adapter,
    attach_action_adapter,
    clear_action_condition,
    stage_sample_condition,
)
from uniaction.model.attention import AttentionBackend, AttentionBackendRuntime
from uniaction.model.lora import (
    inject_lora,
    stable_reinitialize_lora,
    load_and_merge_stage1,
    inject_action_branch_lora,
)
from uniaction.model.weight_cache import cache_files
from uniaction.model.wan import (
    load_wan_pipeline,
    configure_wan_primitive_backends,
    configure_wan_regional_compile,
    iter_wan_dits,
    model_fn_uniaction,
)
from .config import TrainingConfig


class UniActionTrainingModule(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        stage: int,
        *,
        attention_backend: AttentionBackend = "auto",
        attention_fallback: bool = True,
        rope_backend: str = "baseline",
        rmsnorm_backend: str = "baseline",
        fa4_bias_gradient: str = "eager",
        fa4_key_bias_layout: str = "per_key",
        fa4_split_lse_mix_backend: str = "eager",
        checkpoint_preserve_rng_state: bool = True,
        checkpoint_exempt_blocks: int = 0,
        regional_compile_mode: str = "off",
        load_device: str | torch.device = "cpu",
        local_weight_broadcast: LocalWeightBroadcast | None = None,
        cache_dir: str | Path | None = None,
        empty_prompt_context: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        module_started = time.perf_counter()
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        self.config = config
        self.stage = stage
        if not 0 <= checkpoint_exempt_blocks <= config.model.profile.num_blocks:
            raise ValueError(
                f"checkpoint_exempt_blocks must be in [0,{config.model.profile.num_blocks}], got {checkpoint_exempt_blocks}"
            )
        self.checkpoint_preserve_rng_state = bool(checkpoint_preserve_rng_state)
        self.checkpoint_exempt_blocks = int(checkpoint_exempt_blocks)
        self.attention_runtime = AttentionBackendRuntime(
            requested=attention_backend,
            allow_fallback=attention_fallback,
            require_key_bias=stage == 2,
            fa4_bias_gradient=fa4_bias_gradient,
            fa4_key_bias_layout=fa4_key_bias_layout,
            fa4_split_lse_mix_backend=fa4_split_lse_mix_backend,
            reference_count_learnable=False,
        )
        set_seed(config.seed, device_specific=False)
        pipeline_load_started = time.perf_counter()
        self.pipe = load_wan_pipeline(
            config,
            preprocessors=False,
            device=load_device,
            attention_runtime=self.attention_runtime,
            local_weight_broadcast=local_weight_broadcast,
            cache_dir=cache_dir,
        )
        pipeline_loaded = time.perf_counter()
        self.initialization_phases: dict[str, Any] = {
            "load_device": str(torch.device(load_device)),
            "pipeline_load_seconds": pipeline_loaded - pipeline_load_started,
            "component_loads": dict(getattr(self.pipe, "_uniaction_load_timings", {})),
        }
        primitive_started = time.perf_counter()
        self.primitive_backends = {
            name: configure_wan_primitive_backends(
                dit, rope_backend=rope_backend, rmsnorm_backend=rmsnorm_backend
            )
            for name, dit in iter_wan_dits(self.pipe)
        }
        self.initialization_phases["primitive_configuration_seconds"] = (
            time.perf_counter() - primitive_started
        )
        set_seed(config.seed, device_specific=False)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.requires_grad_(False)
        self.adapter: ActionAdapter | None = None
        self._staged_action_dit: nn.Module | None = None
        self._last_action_token_count: int | None = None
        self.expert_updates = {name: 0 for name in config.model.profile.expert_names}
        stage_setup_started = time.perf_counter()
        if stage == 1:
            initialized = 0
            expected = 0
            expert_setup_seconds = {}
            for expert_index, (name, dit) in enumerate(iter_wan_dits(self.pipe)):
                expert_started = time.perf_counter()
                dit = inject_lora(
                    dit,
                    target_modules=tuple(config.optimization.lora_targets),
                    rank=config.optimization.lora_rank,
                    alpha=config.optimization.lora_alpha,
                    dtype=torch.bfloat16,
                )
                initialized += stable_reinitialize_lora(dit, config.seed + expert_index)
                expected += len(dit.blocks) * 10
                setattr(self.pipe, name, dit)
                expert_setup_seconds[name] = time.perf_counter() - expert_started
            self.initialization_phases["stage1_lora_setup_seconds"] = expert_setup_seconds
            if initialized != expected:
                raise RuntimeError(
                    f"Stage 1 initialized {initialized} LoRA layers, expected {expected}"
                )
        else:
            stage1_checkpoint = config.stage2.stage1_checkpoint
            if stage1_checkpoint is None:
                raise ValueError("Stage 2 requires stage1_checkpoint")
            if cache_dir is not None:
                (stage1_checkpoint,), stage1_staging = cache_files(
                    (stage1_checkpoint,),
                    stage_root=Path(cache_dir),
                    namespace=f"checkpoints/{config.model.family}/stage1",
                )
                self.initialization_phases["checkpoint_cache"] = {"stage1": stage1_staging}
            merge_seconds = {}
            for name, dit in iter_wan_dits(self.pipe):
                merge_started = time.perf_counter()
                dit = load_and_merge_stage1(
                    dit,
                    stage1_checkpoint,
                    rank=config.optimization.lora_rank,
                    alpha=config.optimization.lora_alpha,
                    dtype=torch.bfloat16,
                    prefix=name,
                    local_broadcast=local_weight_broadcast,
                )
                setattr(self.pipe, name, dit)
                merge_seconds[name] = time.perf_counter() - merge_started
            self.initialization_phases["stage1_merge_seconds"] = merge_seconds
            adapter_started = time.perf_counter()
            self.adapter = build_action_adapter(
                config, self.pipe.dit, attention_runtime=self.attention_runtime
            )
            self.initialization_phases["action_adapter_build_seconds"] = (
                time.perf_counter() - adapter_started
            )
            action_lora_seconds = {}
            for expert_index, (name, dit) in enumerate(iter_wan_dits(self.pipe)):
                action_lora_started = time.perf_counter()
                attach_action_adapter(dit, self.adapter, register_adapter=expert_index == 0)
                dit = inject_action_branch_lora(
                    dit,
                    rank=config.optimization.lora_rank,
                    alpha=config.optimization.lora_alpha,
                    dtype=torch.bfloat16,
                    init_seed=config.seed + expert_index,
                )
                setattr(self.pipe, name, dit)
                action_lora_seconds[name] = time.perf_counter() - action_lora_started
            self.initialization_phases["action_lora_setup_seconds"] = action_lora_seconds
            self.pipe.action_adapter = self.adapter
            self.adapter.requires_grad_(True)
            self.adapter.selfattn_action_log_reference_count.requires_grad_(False)
        self.initialization_phases["stage_setup_seconds"] = (
            time.perf_counter() - stage_setup_started
        )
        compile_started = time.perf_counter()
        self.regional_compile = {
            name: configure_wan_regional_compile(
                dit, mode=regional_compile_mode, dynamic=stage == 2
            )
            for name, dit in iter_wan_dits(self.pipe)
        }
        self.initialization_phases["regional_compile_wrap_seconds"] = (
            time.perf_counter() - compile_started
        )
        if not self.checkpoint_preserve_rng_state:
            stochastic_blocks = [
                name
                for dit_name, dit in iter_wan_dits(self.pipe)
                for name, child in dit.blocks.named_modules(prefix=f"{dit_name}.blocks")
                if isinstance(child, nn.Dropout) and child.p > 0
            ]
            if stochastic_blocks:
                raise ValueError(
                    "checkpoint RNG preservation cannot be disabled with active Dropout modules: "
                    + ", ".join(stochastic_blocks[:10])
                )
        stage_config = config.stage1 if stage == 1 else config.stage2
        if stage == 1:
            self._prompt_dropout = config.stage1.prompt_dropout
        else:
            self._prompt_dropout = config.stage2.prompt_dropout
        empty_context = None
        if self._prompt_dropout:
            empty_context = empty_prompt_context
            if empty_context is None:
                path = stage_config.empty_prompt_context or config.stage1.empty_prompt_context
                if path is None or not path.is_file():
                    raise FileNotFoundError(f"empty prompt context is missing: {path}")
                empty_context = load_file(str(path), device="cpu").get("context")
            if empty_context is None or empty_context.shape != (1, 512, 4096):
                raise ValueError(
                    f"empty prompt context must be (1,512,4096), got {(None if empty_context is None else tuple(empty_context.shape))}"
                )
            empty_context = empty_context.to(device=load_device)
        self.register_buffer("_empty_context", empty_context, persistent=False)
        rank = int(os.environ.get("RANK", "0"))
        self._prompt_rng = random.Random(config.seed + 1000003 * rank)
        self.prompt_seen = 0
        self.prompt_dropped = 0
        set_seed(config.seed, device_specific=False)
        self._validate_trainable_specification()
        self.initialization_phases["module_construction_seconds"] = (
            time.perf_counter() - module_started
        )

    def to(self, *args, **kwargs):
        for child in self.children():
            child.to(*args, **kwargs)
        for name, buffer in self._buffers.items():
            if buffer is not None:
                self._buffers[name] = buffer.to(*args, **kwargs)
        return self

    def _validate_trainable_specification(self) -> None:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        profile = self.config.model.profile
        expert_count = len(profile.expert_names)
        if self.stage == 1:
            if not names or any(("lora_" not in name for name in names)):
                raise RuntimeError("Stage 1 must train only shared DiT LoRA tensors")
            expected_tensors = expert_count * profile.num_blocks * 10 * 2
        else:
            if self.adapter is None:
                raise RuntimeError("Stage 2 action adapter is missing")
            adapter_names = {name for name in names if "_action_adapter_ref." in name}
            lora_names = {name for name in names if ".lora_" in name}
            expected_lora = expert_count * profile.num_blocks * 6 * 2
            expected_adapter = self.adapter.expected_trainable_tensor_count
            expected_tensors = expected_lora + expected_adapter
            if len(adapter_names) != expected_adapter or len(lora_names) != expected_lora:
                raise RuntimeError(
                    f"Stage 2 trainable schema mismatch: adapter={len(adapter_names)}, lora={len(lora_names)}"
                )
            fp32_names = {
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad and parameter.dtype == torch.float32
            }
            expected_fp32 = 4
            if len(fp32_names) != expected_fp32:
                raise RuntimeError(
                    f"Stage 2 requires four FP32 RMSNorm weights, got {sorted(fp32_names)}"
                )
        if len(names) != expected_tensors:
            raise RuntimeError(
                f"Stage {self.stage} has {len(names)} trainable tensors, expected {expected_tensors}"
            )

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def resolve_attention_backend(self, device: torch.device | str) -> str:
        return self.attention_runtime.resolve(device)

    def attention_backend_summary(self) -> dict[str, object]:
        return self.attention_runtime.summary()

    def clear_staged_action(self) -> None:
        if self._staged_action_dit is not None:
            clear_action_condition(self._staged_action_dit)
            self._staged_action_dit = None

    def staged_action_token_count(self) -> int | None:
        if self._staged_action_dit is None:
            return self._last_action_token_count
        state = getattr(self._staged_action_dit, "_action_state", None)
        if state is None:
            return self._last_action_token_count
        return int(state.cond_size)

    def prompt_dropout_summary(self) -> dict[str, float | int]:
        return {
            "seen": self.prompt_seen,
            "dropped": self.prompt_dropped,
            "fraction": self.prompt_dropped / self.prompt_seen if self.prompt_seen else 0.0,
        }

    def _drop_prompt(self) -> bool:
        dropped = self._prompt_dropout > 0 and self._prompt_rng.random() < self._prompt_dropout
        self.prompt_seen += 1
        self.prompt_dropped += int(dropped)
        return dropped

    def _context(self, sample: dict[str, Any]) -> torch.Tensor:
        source = self._empty_context if self._drop_prompt() else sample["precomputed_context"]
        if source is None:
            raise RuntimeError("Empty prompt context is unavailable")
        return source.to(device=self.pipe.device, dtype=self.pipe.torch_dtype, non_blocking=True)

    def forward(self, sample: dict[str, Any]) -> torch.Tensor:
        input_latents = sample["precomputed_input_latents"].to(
            device=self.pipe.device, dtype=self.pipe.torch_dtype, non_blocking=True
        )
        context = self._context(sample)
        noise = self.pipe.generate_noise(
            tuple(input_latents.shape), seed=None, rand_device=self.pipe.device
        )
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        noisy_latents = self.pipe.scheduler.add_noise(input_latents, noise, timestep)
        target = self.pipe.scheduler.training_target(input_latents, noise, timestep)
        dit = self.pipe.dit
        self.expert_updates["dit"] += 1
        if self.stage == 2:
            assert self.adapter is not None
            stage_sample_condition(
                dit, self.adapter, sample, device=self.pipe.device, dtype=self.pipe.torch_dtype
            )
            self._staged_action_dit = dit
            self._last_action_token_count = self.staged_action_token_count()
        try:
            y = sample["precomputed_y"].to(
                device=self.pipe.device, dtype=self.pipe.torch_dtype, non_blocking=True
            )
            prediction = model_fn_uniaction(
                dit=dit,
                latents=noisy_latents,
                timestep=timestep,
                context=context,
                y=y,
                use_gradient_checkpointing=True,
                checkpoint_preserve_rng_state=self.checkpoint_preserve_rng_state,
                checkpoint_exempt_blocks=self.checkpoint_exempt_blocks,
            )
        except BaseException:
            self.clear_staged_action()
            raise
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        if loss.ndim != 0 or not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError(f"diffusion loss must be one finite scalar, got {loss}")
        return loss * self.pipe.scheduler.training_weight(timestep)
