from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from PIL import Image
from tqdm import tqdm

from uniaction.config import ExperimentConfig
from uniaction.model.action import (
    attach_action_adapter,
    build_action_adapter,
    clear_action_condition,
    stage_sample_condition,
    suspend_action_condition,
)
from uniaction.model.lora import (
    disable_lora_adapters,
    inject_action_branch_lora,
    load_and_merge_stage1,
    load_dit_checkpoint,
)
from uniaction.model.weight_cache import (
    WeightCacheRequest,
    cache_file_groups,
)
from uniaction.model.attention import (
    AttentionBackend,
    AttentionBackendRuntime,
    FA4KeyBiasLayout,
    FA4SplitLSEMixBackend,
)
from uniaction.model.wan import (
    configure_wan_primitive_backends,
    iter_wan_dits,
    load_wan_pipeline,
)
from uniaction.runtime_profiles import (
    resolve_runtime_profile,
    resolve_profile_overrides,
    validate_runtime_summary,
)


CFGMode = Literal["shared_action", "joint", "separate"]


def combine_cfg_predictions(
    action_positive: torch.Tensor,
    *,
    mode: CFGMode,
    text_scale: float,
    action_scale: float,
    base_positive: torch.Tensor | None = None,
    base_negative: torch.Tensor | None = None,
) -> torch.Tensor:

    if text_scale <= 0 or action_scale <= 0:
        raise ValueError("CFG scales must be positive")
    if mode == "joint":
        if base_negative is None:
            raise ValueError("joint CFG requires the empty-text no-action prediction")
        if not math.isclose(text_scale, action_scale):
            raise ValueError("joint CFG requires equal text and action scales")
        return base_negative + text_scale * (action_positive - base_negative)
    if mode == "separate":
        if text_scale == action_scale:
            if text_scale == 1.0:
                return action_positive
            if base_negative is None:
                raise ValueError(
                    "equal-scale separate CFG requires the empty-text no-action prediction"
                )
            return base_negative + text_scale * (action_positive - base_negative)
        if base_positive is None:
            raise ValueError("separate CFG requires the text-only no-action prediction")
        prediction = base_positive + action_scale * (action_positive - base_positive)
        if not math.isclose(text_scale, 1.0):
            if base_negative is None:
                raise ValueError("separate text CFG requires the empty-text no-action prediction")
            prediction = prediction + (text_scale - 1.0) * (base_positive - base_negative)
        return prediction
    raise ValueError(f"explicit CFG does not support mode {mode!r}")


def _base_model_prediction(pipe, dit, inputs: dict[str, Any]) -> torch.Tensor:

    with suspend_action_condition(dit), disable_lora_adapters(dit):
        return pipe.model_fn(dit=dit, **inputs)


def _generate_explicit_cfg(
    config: ExperimentConfig,
    pipe,
    first_frame: Image.Image,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    mode: Literal["joint", "separate"],
    text_scale: float,
    action_scale: float,
    prediction_builder: Callable | None = None,
) -> list[Image.Image]:
    if pipe.dit is None or pipe.vae is None or pipe.text_encoder is None or pipe.tokenizer is None:
        raise RuntimeError("inference requires DiT, VAE, text encoder, and tokenizer")
    equal_scale_separate = mode == "separate" and text_scale == action_scale
    need_negative = mode == "joint" or (
        text_scale != 1.0 if prediction_builder is not None else not math.isclose(text_scale, 1.0)
    )
    pipe.scheduler.set_timesteps(
        config.evaluation.steps,
        denoising_strength=1.0,
        shift=config.evaluation.sigma_shift,
        solver=config.evaluation.solver,
    )
    inputs_shared: dict[str, Any] = {
        "input_image": first_frame,
        "input_video": None,
        "height": config.model.height,
        "width": config.model.width,
        "num_frames": config.model.num_frames,
        "seed": seed,
        "rand_device": "cpu",
        "cfg_scale": 2.0 if need_negative else 1.0,
        "tiled": False,
        "tile_size": None,
        "tile_stride": None,
    }
    inputs_positive: dict[str, Any] = {"prompt": prompt}
    inputs_negative: dict[str, Any] = {"negative_prompt": negative_prompt}
    for unit in pipe.units:
        inputs_shared, inputs_positive, inputs_negative = pipe.unit_runner(
            unit,
            pipe,
            inputs_shared,
            inputs_positive,
            inputs_negative,
        )

    active_dit = pipe.dit
    for step, scheduler_timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
        if (
            scheduler_timestep.item() < config.model.switch_dit_boundary * 1000
            and pipe.dit2 is not None
        ):
            active_dit = pipe.dit2
        timestep = scheduler_timestep.unsqueeze(0).to(
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        positive_inputs = {**inputs_shared, **inputs_positive, "timestep": timestep}
        action_positive = pipe.model_fn(dit=active_dit, **positive_inputs)
        if prediction_builder is not None:
            negative_inputs = (
                {**inputs_shared, **inputs_negative, "timestep": timestep}
                if need_negative
                else None
            )
            prediction = prediction_builder(
                action_positive,
                active_dit,
                positive_inputs,
                negative_inputs,
            )
        elif mode == "joint" or equal_scale_separate:
            negative_inputs = {**inputs_shared, **inputs_negative, "timestep": timestep}
            prediction = combine_cfg_predictions(
                action_positive,
                mode=mode,
                text_scale=text_scale,
                action_scale=action_scale,
                base_negative=_base_model_prediction(
                    pipe,
                    active_dit,
                    negative_inputs,
                ),
            )
        else:
            base_positive = _base_model_prediction(
                pipe,
                active_dit,
                positive_inputs,
            )
            base_negative = None
            if need_negative:
                negative_inputs = {
                    **inputs_shared,
                    **inputs_negative,
                    "timestep": timestep,
                }
                base_negative = _base_model_prediction(
                    pipe,
                    active_dit,
                    negative_inputs,
                )
            prediction = combine_cfg_predictions(
                action_positive,
                mode=mode,
                text_scale=text_scale,
                action_scale=action_scale,
                base_positive=base_positive,
                base_negative=base_negative,
            )
        inputs_shared["latents"] = pipe.scheduler.step(
            prediction,
            pipe.scheduler.timesteps[step],
            inputs_shared["latents"],
        )
        if "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

    video = pipe.vae.decode(
        inputs_shared["latents"],
        device=pipe.device,
        tiled=False,
        tile_size=None,
        tile_stride=None,
    )
    return pipe.vae_output_to_video(video)


def build_i2v_pipeline(
    config: ExperimentConfig,
    *,
    stage1_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    attention_backend: AttentionBackend | None = None,
    attention_fallback: bool | None = None,
    rope_backend: str | None = None,
    rmsnorm_backend: str | None = None,
    allow_compatibility_runtime: bool = False,
):

    runtime_profile = None
    if allow_compatibility_runtime:
        runtime = {
            "attention_backend": attention_backend or "auto",
            "attention_fallback": (True if attention_fallback is None else attention_fallback),
            "rope_backend": rope_backend or "baseline",
            "rmsnorm_backend": rmsnorm_backend or "baseline",
        }
    else:
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "the accelerated I2V runtime requires CUDA on an H100, H200, or RTX PRO 6000"
            )
        device_index = (
            torch.cuda.current_device() if resolved_device.index is None else resolved_device.index
        )
        runtime_profile = resolve_runtime_profile(
            stage=1,
            device_name=torch.cuda.get_device_name(device_index),
            compute_capability=torch.cuda.get_device_capability(device_index),
            model_family=config.model.family,
            num_frames=config.model.num_frames,
        )
        overrides = {name: None for name in runtime_profile.training_kwargs()}
        overrides.update(
            {
                "attention_backend": attention_backend,
                "attention_fallback": attention_fallback,
                "rope_backend": rope_backend,
                "rmsnorm_backend": rmsnorm_backend,
            }
        )
        resolved = resolve_profile_overrides(runtime_profile, overrides)
        runtime = {
            name: resolved[name]
            for name in (
                "attention_backend",
                "attention_fallback",
                "rope_backend",
                "rmsnorm_backend",
            )
        }
    attention_runtime = AttentionBackendRuntime(
        requested=runtime["attention_backend"],
        allow_fallback=runtime["attention_fallback"],
        require_key_bias=False,
        reference_count_learnable=False,
    )
    pipe = load_wan_pipeline(
        config,
        preprocessors=True,
        device=device,
        attention_runtime=attention_runtime,
    )
    if stage1_checkpoint is not None:
        stage1 = Path(stage1_checkpoint)
        for name, dit in iter_wan_dits(pipe):
            dit = load_and_merge_stage1(
                dit,
                stage1,
                rank=config.optimization.lora_rank,
                alpha=config.optimization.lora_alpha,
                dtype=torch.bfloat16,
                prefix=name,
            )
            setattr(pipe, name, dit)
    pipe.requires_grad_(False)
    pipe.eval()
    primitive_backends = {
        name: configure_wan_primitive_backends(
            dit,
            rope_backend=runtime["rope_backend"],
            rmsnorm_backend=runtime["rmsnorm_backend"],
        )
        for name, dit in iter_wan_dits(pipe)
    }
    pipe.to(device=device, dtype=torch.bfloat16)
    attention_runtime.resolve(pipe.device)
    pipe.attention_runtime = attention_runtime
    pipe.primitive_backends = primitive_backends
    pipe.runtime_profile = runtime_profile.as_dict() if runtime_profile is not None else None
    pipe.runtime_specification_mode = (
        "accelerated" if runtime_profile is not None else "explicit_compatibility"
    )
    if runtime_profile is not None:
        validate_runtime_summary(
            runtime_profile,
            attention=attention_runtime.summary(),
            primitive_backends=primitive_backends,
            regional_compile=None,
            require_call_coverage=False,
        )
    return pipe


def _resolve_action_rope_mode(config, stage2: Path):
    from uniaction.checkpoints import read_release

    metadata = read_release(stage2.parent, trained=config.artifacts.trained)
    if "cell_center" != metadata["action_rope_mode"]:
        raise ValueError("Action position mapping does not match the checkpoint")
    return config


def _resolve_action_reads_video(config, stage2: Path):
    from uniaction.checkpoints import read_release

    metadata = read_release(stage2.parent, trained=config.artifacts.trained)
    if True != metadata["action_reads_video"]:
        raise ValueError("Action attention wiring does not match the checkpoint")
    return config


def _stage_checkpoints(
    config: ExperimentConfig,
    *,
    stage1: Path,
    stage2: Path,
    stage_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:

    staged = cache_file_groups(
        {
            "stage1": WeightCacheRequest(
                source_files=(stage1,),
                namespace=f"checkpoints/{config.model.family}/stage1",
            ),
            "stage2": WeightCacheRequest(
                source_files=(stage2,),
                namespace=f"checkpoints/{config.model.family}/stage2",
            ),
        },
        stage_root=stage_root,
    )
    (staged_stage1,) = staged["stage1"][0]
    (staged_stage2,) = staged["stage2"][0]
    return (
        staged_stage1,
        staged_stage2,
        {
            "stage1": staged["stage1"][1],
            "stage2": staged["stage2"][1],
        },
    )


def _expected_stage2_loaded_tensors(config, adapter):
    return config.model.profile.expected_checkpoint_tensors(2)


def build_pipeline(
    config: ExperimentConfig,
    *,
    stage1_checkpoint: str | Path | None = None,
    stage2_checkpoint: str | Path | None = None,
    expected_n_ref: float | None = None,
    device: str | torch.device = "cpu",
    attention_backend: AttentionBackend | None = None,
    attention_fallback: bool | None = None,
    fa4_bias_gradient: str | None = None,
    fa4_key_bias_layout: FA4KeyBiasLayout | None = None,
    fa4_split_lse_mix_backend: FA4SplitLSEMixBackend | None = None,
    rope_backend: str | None = None,
    rmsnorm_backend: str | None = None,
    allow_compatibility_runtime: bool = False,
    action_adapter_builder=build_action_adapter,
    cache_dir: str | Path | None = None,
):
    initialization_started = time.perf_counter()
    from uniaction.checkpoints import verify_checkpoints

    config.validate()
    stage1 = Path(stage1_checkpoint or config.artifacts.stage1_checkpoint)
    stage2 = Path(stage2_checkpoint or config.artifacts.stage2_checkpoint)
    verify_checkpoints(stage1, stage2, trained=config.artifacts.trained)
    if expected_n_ref is None and stage2_checkpoint is None:
        expected_n_ref = config.artifacts.stage2_n_ref
    config = _resolve_action_rope_mode(config, stage2)
    config = _resolve_action_reads_video(config, stage2)
    checkpoint_staging: dict[str, Any] = {}
    stage_root = None if cache_dir is None else Path(cache_dir).expanduser()
    runtime_profile = None
    if allow_compatibility_runtime:
        runtime = {
            "attention_backend": attention_backend or "auto",
            "attention_fallback": (True if attention_fallback is None else attention_fallback),
            "fa4_bias_gradient": fa4_bias_gradient or "eager",
            "fa4_key_bias_layout": fa4_key_bias_layout or "per_key",
            "fa4_split_lse_mix_backend": fa4_split_lse_mix_backend or "eager",
            "rope_backend": rope_backend or "baseline",
            "rmsnorm_backend": rmsnorm_backend or "baseline",
        }
    else:
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPU inference requires CUDA")
        device_index = (
            torch.cuda.current_device() if resolved_device.index is None else resolved_device.index
        )
        runtime_profile = resolve_runtime_profile(
            stage=2,
            device_name=torch.cuda.get_device_name(device_index),
            compute_capability=torch.cuda.get_device_capability(device_index),
            model_family=config.model.family,
            num_frames=config.model.num_frames,
        )
        all_overrides = {name: None for name in runtime_profile.training_kwargs()}
        all_overrides.update(
            {
                "attention_backend": attention_backend,
                "attention_fallback": attention_fallback,
                "fa4_bias_gradient": fa4_bias_gradient,
                "fa4_key_bias_layout": fa4_key_bias_layout,
                "fa4_split_lse_mix_backend": fa4_split_lse_mix_backend,
                "rope_backend": rope_backend,
                "rmsnorm_backend": rmsnorm_backend,
            }
        )
        resolved = resolve_profile_overrides(runtime_profile, all_overrides)
        runtime = {
            name: resolved[name]
            for name in (
                "attention_backend",
                "attention_fallback",
                "fa4_bias_gradient",
                "fa4_key_bias_layout",
                "fa4_split_lse_mix_backend",
                "rope_backend",
                "rmsnorm_backend",
            )
        }
    attention_runtime = AttentionBackendRuntime(
        requested=runtime["attention_backend"],
        allow_fallback=runtime["attention_fallback"],
        require_key_bias=True,
        fa4_bias_gradient=runtime["fa4_bias_gradient"],
        fa4_key_bias_layout=runtime["fa4_key_bias_layout"],
        fa4_split_lse_mix_backend=runtime["fa4_split_lse_mix_backend"],
        reference_count_learnable=False,
    )
    checkpoint_stage_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="uniaction-checkpoint-stage")
        if stage_root is not None
        else None
    )
    checkpoint_stage_future = (
        checkpoint_stage_executor.submit(
            _stage_checkpoints,
            config,
            stage1=stage1,
            stage2=stage2,
            stage_root=stage_root,
        )
        if checkpoint_stage_executor is not None and stage_root is not None
        else None
    )
    pipeline_load_started = time.perf_counter()
    try:
        pipe = load_wan_pipeline(
            config,
            preprocessors=True,
            device=device,
            attention_runtime=attention_runtime,
            cache_dir=stage_root,
        )
        if checkpoint_stage_future is not None:
            stage1, stage2, checkpoint_staging = checkpoint_stage_future.result()
    finally:
        if checkpoint_stage_executor is not None:
            checkpoint_stage_executor.shutdown(wait=True)
    pipeline_loaded = time.perf_counter()
    initialization_phases: dict[str, Any] = {
        "load_device": str(torch.device(device)),
        "pipeline_load_seconds": pipeline_loaded - pipeline_load_started,
        "component_loads": dict(getattr(pipe, "_uniaction_load_timings", {})),
        "checkpoint_cache": checkpoint_staging,
    }
    stage1_merge_seconds = {}
    for name, dit in iter_wan_dits(pipe):
        merge_started = time.perf_counter()
        dit = load_and_merge_stage1(
            dit,
            stage1,
            rank=config.optimization.lora_rank,
            alpha=config.optimization.lora_alpha,
            dtype=torch.bfloat16,
            prefix=name,
        )
        setattr(pipe, name, dit)
        stage1_merge_seconds[name] = time.perf_counter() - merge_started
    initialization_phases["stage1_merge_seconds"] = stage1_merge_seconds
    adapter_started = time.perf_counter()
    adapter = action_adapter_builder(
        config,
        pipe.dit,
        attention_runtime=attention_runtime,
    )
    initialization_phases["action_adapter_build_seconds"] = time.perf_counter() - adapter_started
    action_lora_seconds = {}
    for expert_index, (name, dit) in enumerate(iter_wan_dits(pipe)):
        action_lora_started = time.perf_counter()
        attach_action_adapter(
            dit,
            adapter,
            register_adapter=expert_index == 0,
        )
        dit = inject_action_branch_lora(
            dit,
            rank=config.optimization.lora_rank,
            alpha=config.optimization.lora_alpha,
            dtype=torch.bfloat16,
        )
        setattr(pipe, name, dit)
        action_lora_seconds[name] = time.perf_counter() - action_lora_started
    initialization_phases["action_lora_setup_seconds"] = action_lora_seconds
    pipe.action_adapter = adapter

    pipe.action_rope_mode = "cell_center"
    primitive_started = time.perf_counter()
    primitive_backends = {
        name: configure_wan_primitive_backends(
            dit,
            rope_backend=runtime["rope_backend"],
            rmsnorm_backend=runtime["rmsnorm_backend"],
        )
        for name, dit in iter_wan_dits(pipe)
    }
    initialization_phases["primitive_configuration_seconds"] = (
        time.perf_counter() - primitive_started
    )
    loaded = 0
    stage2_load_seconds = {}
    for expert_index, (name, dit) in enumerate(iter_wan_dits(pipe)):
        checkpoint_load_started = time.perf_counter()
        loaded += load_dit_checkpoint(
            dit,
            stage2,
            require_adapter=expert_index == 0,
            prefix=name,
        )
        stage2_load_seconds[name] = time.perf_counter() - checkpoint_load_started
    initialization_phases["stage2_checkpoint_load_seconds"] = stage2_load_seconds
    expected = _expected_stage2_loaded_tensors(config, adapter)
    if loaded != expected:
        raise RuntimeError(f"Stage-2 checkpoint must contain {expected} tensors, got {loaded}")
    effective_n_ref = math.exp(float(adapter.selfattn_action_log_reference_count.detach().float()))
    if expected_n_ref is not None and not math.isclose(
        effective_n_ref, expected_n_ref, rel_tol=1e-6
    ):
        raise RuntimeError(f"loaded N_ref={effective_n_ref} != expected {expected_n_ref}")
    finalize_started = time.perf_counter()
    pipe.requires_grad_(False)
    pipe.eval()
    pipe.to(device=device, dtype=torch.bfloat16)
    attention_runtime.resolve(pipe.device)
    pipe.attention_runtime = attention_runtime
    pipe.primitive_backends = primitive_backends
    pipe.runtime_profile = runtime_profile.as_dict() if runtime_profile is not None else None
    pipe.runtime_specification_mode = (
        "accelerated" if runtime_profile is not None else "explicit_compatibility"
    )
    if runtime_profile is not None:
        validate_runtime_summary(
            runtime_profile,
            attention=attention_runtime.summary(),
            primitive_backends=primitive_backends,
            regional_compile=None,
            require_call_coverage=False,
        )
    initialization_phases["finalize_seconds"] = time.perf_counter() - finalize_started
    initialization_phases["total_seconds"] = time.perf_counter() - initialization_started
    pipe._uniaction_initialization_phases = initialization_phases
    return pipe, adapter


@torch.no_grad()
def generate_i2v_sample(
    config: ExperimentConfig,
    pipe,
    first_frame: Image.Image,
    *,
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
) -> list[Image.Image]:

    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=first_frame,
        height=config.model.height,
        width=config.model.width,
        num_frames=config.model.num_frames,
        seed=config.evaluation.seed if seed is None else int(seed),
        cfg_scale=config.evaluation.cfg_scale,
        num_inference_steps=config.evaluation.steps,
        sigma_shift=config.evaluation.sigma_shift,
        sample_solver=config.evaluation.solver,
        switch_DiT_boundary=config.model.switch_dit_boundary,
        tiled=False,
    )


@torch.no_grad()
def generate_sample(
    config: ExperimentConfig,
    pipe,
    adapter,
    sample: dict[str, Any],
    first_frame: Image.Image,
    *,
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
    cfg_mode: CFGMode = "shared_action",
    text_cfg_scale: float | None = None,
    action_cfg_scale: float | None = None,
) -> list[Image.Image]:
    if cfg_mode not in ("shared_action", "joint", "separate"):
        raise ValueError(f"unsupported cfg_mode={cfg_mode!r}")
    text_scale = config.evaluation.cfg_scale if text_cfg_scale is None else float(text_cfg_scale)
    action_scale = (
        (text_scale if cfg_mode == "joint" else 1.0)
        if action_cfg_scale is None
        else float(action_cfg_scale)
    )
    if text_scale <= 0 or action_scale <= 0:
        raise ValueError("CFG scales must be positive")
    if cfg_mode == "joint" and not math.isclose(text_scale, action_scale):
        raise ValueError("joint CFG requires equal text and action scales")
    for _, dit in iter_wan_dits(pipe):
        stage_sample_condition(
            dit,
            adapter,
            sample,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
    try:
        effective_seed = config.evaluation.seed if seed is None else int(seed)
        if cfg_mode == "shared_action":
            frames = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=first_frame,
                height=config.model.height,
                width=config.model.width,
                num_frames=config.model.num_frames,
                seed=effective_seed,
                cfg_scale=text_scale,
                num_inference_steps=config.evaluation.steps,
                sigma_shift=config.evaluation.sigma_shift,
                sample_solver=config.evaluation.solver,
                switch_DiT_boundary=config.model.switch_dit_boundary,
                tiled=False,
            )
        elif math.isclose(text_scale, 1.0) and math.isclose(action_scale, 1.0):
            frames = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=first_frame,
                height=config.model.height,
                width=config.model.width,
                num_frames=config.model.num_frames,
                seed=effective_seed,
                cfg_scale=1.0,
                num_inference_steps=config.evaluation.steps,
                sigma_shift=config.evaluation.sigma_shift,
                sample_solver=config.evaluation.solver,
                switch_DiT_boundary=config.model.switch_dit_boundary,
                tiled=False,
            )
        else:
            frames = _generate_explicit_cfg(
                config,
                pipe,
                first_frame,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=effective_seed,
                mode=cfg_mode,
                text_scale=text_scale,
                action_scale=action_scale,
            )
    finally:
        for _, dit in iter_wan_dits(pipe):
            clear_action_condition(dit)
    return frames
