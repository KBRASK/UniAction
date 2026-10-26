from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import types
from typing import Any

import torch

from uniaction.backbones.wan.loader import (
    LocalWeightBroadcast,
    load_dit,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)
from uniaction.backbones.wan.dit import RMSNorm, SelfAttention
from uniaction.backbones.wan.pipeline import WanPipeline, model_fn_wan_video
from uniaction.config import ExperimentConfig
from uniaction.model.attention import AttentionBackendRuntime, RuntimeAttentionModule
from uniaction.model.weight_cache import (
    WeightCacheRequest,
    cache_file_groups,
)
from uniaction.model.weight_mirror import validate_bf16_mirror
from uniaction.profiles import WanModelProfile


@dataclass(frozen=True)
class WanComponentFiles:
    experts: tuple[tuple[Path, ...], ...]
    t5: Path | None
    vae: Path | None
    image_encoder: Path | None
    expert_mirrors: tuple[dict[str, Any] | None, ...]
    weight_cache: dict[str, dict[str, Any]]


def model_fn_uniaction(
    *,
    dit,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    y: torch.Tensor | None = None,
    clip_feature: torch.Tensor | None = None,
    reference_latents: torch.Tensor | None = None,
    fuse_vae_embedding_in_latents: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    checkpoint_preserve_rng_state: bool = True,
    checkpoint_exempt_blocks: int = 0,
    use_unified_sequence_parallel: bool = False,
    control_camera_latents_input: torch.Tensor | None = None,
    **_: Any,
) -> torch.Tensor:

    if use_unified_sequence_parallel:
        raise ValueError("UniAction does not support sequence parallelism")
    if reference_latents is not None:
        raise ValueError("UniAction does not support reference-latent concatenation")
    if control_camera_latents_input is not None:
        raise ValueError("UniAction does not support camera-control latents")
    if bool(y is not None) != bool(dit.require_vae_embedding):
        raise ValueError(
            f"DiT require_vae_embedding={dit.require_vae_embedding}, but y presence={y is not None}"
        )
    if bool(clip_feature is not None) != bool(dit.require_clip_embedding):
        raise ValueError(
            f"DiT require_clip_embedding={dit.require_clip_embedding}, "
            f"but clip_feature presence={clip_feature is not None}"
        )
    if bool(fuse_vae_embedding_in_latents) != bool(dit.fuse_vae_embedding_in_latents):
        raise ValueError(
            f"DiT fuse_vae_embedding_in_latents={dit.fuse_vae_embedding_in_latents}, "
            f"but call requested {fuse_vae_embedding_in_latents}"
        )
    return model_fn_wan_video(
        dit=dit,
        latents=latents,
        timestep=timestep,
        context=context,
        y=y,
        clip_feature=clip_feature,
        fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        checkpoint_preserve_rng_state=checkpoint_preserve_rng_state,
        checkpoint_exempt_blocks=checkpoint_exempt_blocks,
    )


def _glob_component(root: Path, pattern: str, label: str) -> tuple[Path, ...]:
    files = tuple(sorted(root.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"no {label} files match {root / pattern}")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing model files: " + ", ".join(map(str, missing)))
    return files


def _required_model_files(
    config: ExperimentConfig,
    preprocessors: bool,
    *,
    cache_dir: Path | None = None,
) -> WanComponentFiles:
    model = config.model
    source_primary = _glob_component(model.model_root, model.dit_pattern, "primary DiT")
    primary = source_primary
    primary_mirror = None
    if model.dit_bf16_pattern is not None:
        assert model.dit_bf16_manifest is not None
        primary = _glob_component(
            model.model_root,
            model.dit_bf16_pattern,
            "primary BF16 DiT mirror",
        )
        primary_mirror = validate_bf16_mirror(
            source_primary,
            primary,
            manifest_path=model.dit_bf16_manifest,
        )
    experts = [primary]
    expert_mirrors: list[dict[str, Any] | None] = [primary_mirror]
    if model.dit2_pattern is not None:
        secondary = _glob_component(model.model_root, model.dit2_pattern, "secondary DiT")
        experts.append(secondary)
        expert_mirrors.append(None)

    t5 = model.common_root / model.t5_file if preprocessors else None
    vae = model.common_root / model.vae_file if preprocessors else None
    image_encoder = (
        model.model_root / model.image_encoder_file
        if preprocessors and model.image_encoder_file is not None
        else None
    )
    missing = [path for path in (t5, vae, image_encoder) if path is not None and not path.is_file()]
    if missing:
        raise FileNotFoundError("missing model files: " + ", ".join(map(str, missing)))
    staging: dict[str, dict[str, Any]] = {}
    if cache_dir is not None:
        requests = {
            "dit": WeightCacheRequest(
                source_files=experts[0],
                namespace=f"models/{model.family}/dit",
            )
        }
        if len(experts) == 2:
            requests["dit2"] = WeightCacheRequest(
                source_files=experts[1],
                namespace=f"models/{model.family}/dit2",
            )
        if t5 is not None:
            requests["text_encoder"] = WeightCacheRequest(
                source_files=(t5,),
                namespace=f"models/{model.family}/text_encoder",
            )
        if vae is not None:
            requests["vae"] = WeightCacheRequest(
                source_files=(vae,),
                namespace=f"models/{model.family}/vae",
            )
        if image_encoder is not None:
            requests["image_encoder"] = WeightCacheRequest(
                source_files=(image_encoder,),
                namespace=f"models/{model.family}/image_encoder",
            )
        staged = cache_file_groups(
            requests,
            stage_root=cache_dir,
        )
        experts = [staged["dit"][0]]
        if "dit2" in staged:
            experts.append(staged["dit2"][0])
        if "text_encoder" in staged:
            (t5,) = staged["text_encoder"][0]
        if "vae" in staged:
            (vae,) = staged["vae"][0]
        if "image_encoder" in staged:
            (image_encoder,) = staged["image_encoder"][0]
        staging = {name: result[1] for name, result in staged.items()}
    return WanComponentFiles(
        tuple(experts),
        t5,
        vae,
        image_encoder,
        tuple(expert_mirrors),
        staging,
    )


def validate_wan_training_assets(config: ExperimentConfig) -> dict[str, Any]:

    components = _required_model_files(config, preprocessors=False)
    expert_files = [path for expert in components.experts for path in expert]
    mirrors = [mirror for mirror in components.expert_mirrors if mirror is not None]
    return {
        "model_root": str(config.model.model_root),
        "expert_count": len(components.experts),
        "expert_file_count": len(expert_files),
        "expert_bytes": sum(path.stat().st_size for path in expert_files),
        "expert_files": [str(path) for path in expert_files],
        "verified_mirrors": mirrors,
    }


def _validate_preprocessors(pipe: WanPipeline, profile: WanModelProfile) -> None:
    missing = [
        name for name in ("text_encoder", "vae", "tokenizer") if getattr(pipe, name, None) is None
    ]
    if profile.requires_clip and getattr(pipe, "image_encoder", None) is None:
        missing.append("image_encoder")
    if not profile.requires_clip and getattr(pipe, "image_encoder", None) is not None:
        raise RuntimeError(f"{profile.family} unexpectedly loaded an image encoder")
    if missing:
        raise RuntimeError("full pipeline is missing preprocessors: " + ", ".join(missing))


def iter_wan_dits(pipe: WanPipeline) -> tuple[tuple[str, torch.nn.Module], ...]:
    if pipe.dit is None:
        raise RuntimeError("Wan pipeline has no primary DiT")
    experts: list[tuple[str, torch.nn.Module]] = [("dit", pipe.dit)]
    if pipe.dit2 is not None:
        experts.append(("dit2", pipe.dit2))
    return tuple(experts)


def configure_wan_primitive_backends(
    dit: torch.nn.Module,
    *,
    rope_backend: str,
    rmsnorm_backend: str,
) -> dict[str, int | str]:

    if rope_backend not in {"baseline", "quack"}:
        raise ValueError(f"unsupported Wan RoPE backend {rope_backend!r}")
    if rmsnorm_backend not in {"baseline", "quack"}:
        raise ValueError(f"unsupported Wan RMSNorm backend {rmsnorm_backend!r}")
    self_attention_count = 0
    rmsnorm_count = 0
    for module in dit.modules():
        if isinstance(module, SelfAttention):
            module.rope_backend = rope_backend
            self_attention_count += 1
        if isinstance(module, RMSNorm):
            module.backend = rmsnorm_backend
            rmsnorm_count += 1
    if self_attention_count != len(dit.blocks):
        raise RuntimeError(
            f"configured {self_attention_count} Wan self-attention modules, "
            f"expected {len(dit.blocks)}"
        )
    if rmsnorm_count == 0:
        raise RuntimeError("loaded Wan DiT contains no configurable RMSNorm modules")
    dit._uniaction_rope_backend = rope_backend
    dit._uniaction_rmsnorm_backend = rmsnorm_backend
    return {
        "rope_backend": rope_backend,
        "rmsnorm_backend": rmsnorm_backend,
        "self_attention_modules": self_attention_count,
        "rmsnorm_modules": rmsnorm_count,
    }


def configure_wan_regional_compile(
    dit: torch.nn.Module,
    *,
    mode: str,
    dynamic: bool,
) -> dict[str, int | str | bool]:

    supported_modes = {
        "off",
        "default",
        "reduce-overhead",
        "max-autotune-no-cudagraphs",
    }
    if mode not in supported_modes:
        raise ValueError(f"unsupported Wan regional compile mode {mode!r}")
    if mode == "off":
        return {
            "mode": mode,
            "dynamic": bool(dynamic),
            "compiled_blocks": 0,
            "attention_graph_boundary": False,
        }
    attention_runtime = getattr(dit, "_uniaction_attention_runtime", None)
    if attention_runtime is not None:
        attention_runtime.install_dynamo_boundary()
    for block in dit.blocks:
        block.compile(
            fullgraph=False,
            dynamic=bool(dynamic),
            mode=mode,
        )
    return {
        "mode": mode,
        "dynamic": bool(dynamic),
        "compiled_blocks": len(dit.blocks),
        "attention_graph_boundary": attention_runtime is not None,
    }


def select_training_dit(
    pipe: WanPipeline,
    timestep: torch.Tensor,
    switch_dit_boundary: float,
) -> tuple[str, torch.nn.Module]:
    if pipe.dit2 is None:
        return "dit", pipe.dit
    if timestep.numel() != 1:
        raise ValueError("dual-DiT training requires one timestep per update")
    if float(timestep.detach().item()) < float(switch_dit_boundary) * 1000.0:
        return "dit2", pipe.dit2
    return "dit", pipe.dit


def _validate_loaded_dit(dit: torch.nn.Module, profile: WanModelProfile, name: str) -> None:
    expected = {
        "dim": profile.hidden_dim,
        "in_dim": profile.in_dim,
    }
    for attribute, value in expected.items():
        actual = getattr(dit, attribute, None)
        if actual != value:
            raise RuntimeError(f"{name}.{attribute}={actual}, expected {value}")
    if len(dit.blocks) != profile.num_blocks:
        raise RuntimeError(f"{name} has {len(dit.blocks)} blocks, expected {profile.num_blocks}")
    heads = getattr(dit.blocks[0].self_attn, "num_heads", None)
    if heads != profile.num_heads:
        raise RuntimeError(f"{name} has {heads} attention heads, expected {profile.num_heads}")
    expected_vae = profile.conditioning == "i2v_y"
    flags = {
        "require_vae_embedding": expected_vae,
        "require_clip_embedding": profile.requires_clip,
        "fuse_vae_embedding_in_latents": profile.conditioning == "fused_first_frame",
    }
    for attribute, value in flags.items():
        actual = bool(getattr(dit, attribute, None))
        if actual != value:
            raise RuntimeError(f"{name}.{attribute}={actual}, expected {value}")


def _runtime_image_cross_attention_forward(
    self: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:

    img = y[:, :257]
    context = y[:, 257:]
    query = self.norm_q(self.q(x))
    key = self.norm_k(self.k(context))
    value = self.v(context)
    output = self.attn(query, key, value)
    image_key = self.norm_k_img(self.k_img(img))
    image_value = self.v_img(img)
    output = output + self.attn(query, image_key, image_value)
    return self.o(output)


def attach_wan_attention_runtime(
    dit: torch.nn.Module,
    runtime: AttentionBackendRuntime,
) -> None:

    attached = getattr(dit, "_uniaction_attention_runtime", None)
    if attached is not None:
        if attached is not runtime:
            raise RuntimeError("a different attention runtime is already attached")
        return
    for block in dit.blocks:
        block.self_attn.attn = RuntimeAttentionModule(
            runtime,
            block.self_attn.num_heads,
        )
        block.cross_attn.attn = RuntimeAttentionModule(
            runtime,
            block.cross_attn.num_heads,
        )
        if block.cross_attn.has_image_input:
            block.cross_attn.forward = types.MethodType(
                _runtime_image_cross_attention_forward,
                block.cross_attn,
            )
    object.__setattr__(dit, "_uniaction_attention_runtime", runtime)


def load_wan_pipeline(
    config: ExperimentConfig,
    *,
    preprocessors: bool,
    device: str | torch.device = "cpu",
    attention_runtime: AttentionBackendRuntime | None = None,
    local_weight_broadcast: LocalWeightBroadcast | None = None,
    cache_dir: str | Path | None = None,
) -> WanPipeline:
    profile = config.model.profile
    stage_root = None if cache_dir is None else Path(cache_dir).expanduser()
    files = _required_model_files(
        config,
        preprocessors,
        cache_dir=stage_root,
    )
    pipe = WanPipeline(torch_dtype=torch.bfloat16, device=device)
    load_timings: dict[str, dict[str, Any]] = {}
    for index, expert_files in enumerate(files.experts):
        name = "dit" if index == 0 else "dit2"
        dit = load_dit(
            profile,
            expert_files,
            dtype=torch.bfloat16,
            device=device,
            local_broadcast=local_weight_broadcast,
        )
        setattr(
            pipe,
            name,
            dit,
        )
        load_timings[name] = dict(getattr(dit, "_uniaction_load_timing", {}))
        if files.expert_mirrors[index] is not None:
            load_timings[name]["bf16_mirror"] = files.expert_mirrors[index]
        if name in files.weight_cache:
            load_timings[name]["weight_cache"] = files.weight_cache[name]
    if preprocessors:
        assert files.t5 is not None and files.vae is not None
        pipe.text_encoder = load_text_encoder(
            files.t5,
            dtype=torch.bfloat16,
            device=device,
        )
        load_timings["text_encoder"] = dict(
            getattr(pipe.text_encoder, "_uniaction_load_timing", {})
        )
        if "text_encoder" in files.weight_cache:
            load_timings["text_encoder"]["weight_cache"] = files.weight_cache["text_encoder"]
        pipe.vae = load_vae(
            profile,
            files.vae,
            dtype=torch.bfloat16,
            device=device,
        )
        load_timings["vae"] = dict(getattr(pipe.vae, "_uniaction_load_timing", {}))
        if "vae" in files.weight_cache:
            load_timings["vae"]["weight_cache"] = files.weight_cache["vae"]
        pipe.tokenizer = load_tokenizer(config.model.tokenizer_root)
        pipe.height_division_factor = pipe.vae.upsampling_factor * 2
        pipe.width_division_factor = pipe.vae.upsampling_factor * 2
    loaded_experts = iter_wan_dits(pipe)
    if len(loaded_experts) != len(profile.expert_names):
        raise RuntimeError(
            f"{profile.family} loaded {len(loaded_experts)} DiTs, "
            f"expected {len(profile.expert_names)}"
        )
    for (actual_name, dit), expected_name in zip(loaded_experts, profile.expert_names):
        if actual_name != expected_name:
            raise RuntimeError(f"loaded expert order {actual_name}, expected {expected_name}")
        _validate_loaded_dit(dit, profile, actual_name)
        if attention_runtime is not None:
            attach_wan_attention_runtime(dit, attention_runtime)
    if preprocessors:
        _validate_preprocessors(pipe, profile)
    else:
        unexpected = [
            name
            for name in ("text_encoder", "vae", "image_encoder", "tokenizer")
            if getattr(pipe, name, None) is not None
        ]
        if unexpected:
            raise RuntimeError("DiT-only loading unexpectedly created: " + ", ".join(unexpected))
    pipe.model_fn = model_fn_uniaction
    pipe._uniaction_load_timings = load_timings
    return pipe
