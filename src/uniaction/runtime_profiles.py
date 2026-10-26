from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import importlib.util
from typing import Any


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    gpu_family: str
    stage: int
    compute_capability: tuple[int, int]
    attention_backend: str
    attention_fallback: bool
    rope_backend: str
    rmsnorm_backend: str
    fa4_bias_gradient: str
    fa4_key_bias_layout: str
    fa4_split_lse_mix_backend: str
    checkpoint_preserve_rng_state: bool
    checkpoint_exempt_blocks: int
    regional_compile_mode: str
    input_backend: str = "cache"
    gradient_checkpointing: bool = False

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["compute_capability"] = list(self.compute_capability)
        return values

    def training_kwargs(self) -> dict[str, Any]:
        return {
            "attention_backend": self.attention_backend,
            "attention_fallback": self.attention_fallback,
            "rope_backend": self.rope_backend,
            "rmsnorm_backend": self.rmsnorm_backend,
            "fa4_bias_gradient": self.fa4_bias_gradient,
            "fa4_key_bias_layout": self.fa4_key_bias_layout,
            "fa4_split_lse_mix_backend": self.fa4_split_lse_mix_backend,
            "checkpoint_preserve_rng_state": self.checkpoint_preserve_rng_state,
            "checkpoint_exempt_blocks": self.checkpoint_exempt_blocks,
            "regional_compile_mode": self.regional_compile_mode,
        }


_H200_COMMON = {
    "gpu_family": "h200",
    "compute_capability": (9, 0),
    "attention_backend": "fa4",
    "attention_fallback": True,
    "rope_backend": "quack",
    "rmsnorm_backend": "quack",
    "fa4_bias_gradient": "deferred",
    "fa4_key_bias_layout": "split_lse",
    "fa4_split_lse_mix_backend": "inductor",
}

_H100_COMMON = {
    **_H200_COMMON,
    "gpu_family": "h100",
}

_PRO6000_COMMON = {
    "gpu_family": "pro6000",
    "compute_capability": (12, 0),
    "attention_backend": "fa4",
    "attention_fallback": True,
    "rope_backend": "baseline",
    "rmsnorm_backend": "baseline",
    "fa4_bias_gradient": "deferred",
    "fa4_key_bias_layout": "split_lse",
    "fa4_split_lse_mix_backend": "inductor",
}

_PROFILES = {
    ("h100", 1): RuntimeProfile(
        name="h100_stage1",
        stage=1,
        checkpoint_preserve_rng_state=False,
        checkpoint_exempt_blocks=2,
        regional_compile_mode="default",
        **_H100_COMMON,
    ),
    ("h100", 2): RuntimeProfile(
        name="h100_stage2",
        stage=2,
        checkpoint_preserve_rng_state=False,
        checkpoint_exempt_blocks=4,
        regional_compile_mode="off",
        **_H100_COMMON,
    ),
    ("h200", 1): RuntimeProfile(
        name="h200_stage1",
        stage=1,
        checkpoint_preserve_rng_state=False,
        checkpoint_exempt_blocks=2,
        regional_compile_mode="default",
        **_H200_COMMON,
    ),
    ("h200", 2): RuntimeProfile(
        name="h200_stage2",
        stage=2,
        checkpoint_preserve_rng_state=False,
        checkpoint_exempt_blocks=4,
        regional_compile_mode="off",
        **_H200_COMMON,
    ),
    ("pro6000", 1): RuntimeProfile(
        name="pro6000_stage1",
        stage=1,
        checkpoint_preserve_rng_state=True,
        checkpoint_exempt_blocks=0,
        regional_compile_mode="default",
        **_PRO6000_COMMON,
    ),
    ("pro6000", 2): RuntimeProfile(
        name="pro6000_stage2",
        stage=2,
        checkpoint_preserve_rng_state=True,
        checkpoint_exempt_blocks=0,
        regional_compile_mode="default",
        **_PRO6000_COMMON,
    ),
}


def _gpu_family(device_name: str) -> str:
    normalized = " ".join(device_name.upper().split())
    if "H100" in normalized:
        return "h100"
    if "H200" in normalized:
        return "h200"
    if "RTX PRO 6000" in normalized:
        return "pro6000"
    return "generic"


def resolve_runtime_profile(
    *,
    stage: int,
    device_name: str,
    compute_capability: tuple[int, int],
    model_family: str | None = None,
    num_frames: int | None = None,
) -> RuntimeProfile:
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")
    family = _gpu_family(device_name)
    observed = tuple(int(value) for value in compute_capability)
    profile = _PROFILES.get((family, stage))
    if profile is None or observed != profile.compute_capability:
        profile = RuntimeProfile(
            name=f"cuda_stage{stage}",
            gpu_family="generic",
            stage=stage,
            compute_capability=observed,
            attention_backend="fa4",
            attention_fallback=True,
            rope_backend="baseline",
            rmsnorm_backend="baseline",
            fa4_bias_gradient="deferred",
            fa4_key_bias_layout="split_lse",
            fa4_split_lse_mix_backend="inductor",
            checkpoint_preserve_rng_state=True,
            checkpoint_exempt_blocks=0,
            regional_compile_mode="off",
        )
    if importlib.util.find_spec("quack") is None:
        profile = replace(profile, rope_backend="baseline", rmsnorm_backend="baseline")
    if (
        model_family
        in {
            "Wan2.2-I2V-A14B",
            "Wan2.2-I2V-A14B-HighNoise",
        }
        and num_frames == 81
    ):
        profile = replace(
            profile,
            name=f"{profile.name}_wan14b_81f",
            gradient_checkpointing=True,
        )
    return profile


def resolve_profile_overrides(
    profile: RuntimeProfile,
    overrides: dict[str, Any],
) -> dict[str, Any]:

    expected = profile.training_kwargs()
    unexpected = set(overrides) - set(expected)
    if unexpected:
        raise KeyError(f"unsupported runtime override keys: {sorted(unexpected)}")
    resolved: dict[str, Any] = {}
    mismatches: dict[str, dict[str, Any]] = {}
    for name, required in expected.items():
        requested = overrides.get(name)
        resolved[name] = required if requested is None else requested
        if resolved[name] != required:
            mismatches[name] = {"required": required, "requested": resolved[name]}
    if mismatches:
        raise RuntimeError(
            f"runtime settings do not match profile {profile.name}: {mismatches}. "
            "Use the profile settings or select the SDPA backend for compatibility."
        )
    return resolved


def validate_runtime_summary(
    profile: RuntimeProfile,
    *,
    attention: dict[str, Any],
    primitive_backends: dict[str, dict[str, Any]],
    regional_compile: dict[str, dict[str, Any]] | None,
    require_call_coverage: bool,
) -> None:
    mismatches: dict[str, Any] = {}
    required_attention = {
        "requested": "fa4",
        "selected": attention.get("selected")
        if attention.get("selected") in {"fa4", "fa3", "fa2", "legacy"}
        else "fa4",
        "fallback_allowed": profile.attention_fallback,
    }
    if profile.stage == 2:
        required_attention.update(
            {
                "fa4_bias_gradient": "deferred",
                "fa4_key_bias_layout": "split_lse",
                "fa4_split_lse_mix_backend": "inductor",
            }
        )
    for name, required in required_attention.items():
        if attention.get(name) != required:
            mismatches[f"attention.{name}"] = {
                "required": required,
                "actual": attention.get(name),
            }
    algorithm = str(attention.get("algorithm", ""))
    if attention.get("selected") == "fa4" and profile.stage == 1 and algorithm != "native_fa4_d128":
        mismatches["attention.algorithm"] = {
            "required": "native_fa4_d128",
            "actual": algorithm,
        }
    if attention.get("selected") == "fa4" and profile.stage == 2:
        required_fragments = (
            "native_fa4_d128",
            "split_plain_fa4_lse_mix_d128",
            "inductor_mix",
        )
        if not all(fragment in algorithm for fragment in required_fragments):
            mismatches["attention.algorithm"] = {
                "required_fragments": list(required_fragments),
                "actual": algorithm,
            }
        if "score_mod" in algorithm or "cutedsl" in algorithm:
            mismatches["attention.unsupported_score_modifier"] = algorithm
    if require_call_coverage:
        if int(attention.get("plain_calls", 0)) <= 0:
            mismatches["attention.plain_calls"] = "must be positive"
        key_bias_calls = int(attention.get("key_bias_calls", 0))
        if profile.stage == 1 and key_bias_calls != 0:
            mismatches["attention.key_bias_calls"] = {
                "required": 0,
                "actual": key_bias_calls,
            }
        if profile.stage == 2 and key_bias_calls <= 0:
            mismatches["attention.key_bias_calls"] = "must be positive"
    if not primitive_backends:
        mismatches["primitive_backends"] = "must not be empty"
    for expert, values in primitive_backends.items():
        for name, required in (
            ("rope_backend", profile.rope_backend),
            ("rmsnorm_backend", profile.rmsnorm_backend),
        ):
            if values.get(name) != required:
                mismatches[f"primitive_backends.{expert}.{name}"] = {
                    "required": required,
                    "actual": values.get(name),
                }
    if regional_compile is not None:
        if not regional_compile:
            mismatches["regional_compile"] = "must not be empty"
        for expert, values in regional_compile.items():
            actual = values.get("mode", "off")
            if actual != profile.regional_compile_mode:
                mismatches[f"regional_compile.{expert}.mode"] = {
                    "required": profile.regional_compile_mode,
                    "actual": actual,
                }
    if mismatches:
        raise RuntimeError(
            f"resolved runtime violates runtime profile {profile.name}: {mismatches}"
        )
