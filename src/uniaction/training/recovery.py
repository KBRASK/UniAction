from __future__ import annotations
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import save_file

RECOVERY_FORMAT = "uniaction-training-recovery-v1"
_RELEASE_PERSISTENT_PARAMETER_SUFFIXES = ("selfattn_action_log_reference_count",)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path(path: Path, suffix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(descriptor)
    return Path(name)


def inventory_file(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size}


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    temporary = _temporary_path(path, ".json.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        inventory = inventory_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return inventory
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, value: Any) -> dict[str, Any]:
    temporary = _temporary_path(path, ".pt.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        inventory = inventory_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return inventory
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    temporary = _temporary_path(path, ".safetensors.tmp")
    try:
        contiguous = {
            name: tensor.detach().to(device="cpu").contiguous() for name, tensor in tensors.items()
        }
        save_file(contiguous, str(temporary))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        inventory = inventory_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return inventory
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def commit_recovery_directory(
    temporary: Path, final: Path, *, latest_payload: dict[str, Any]
) -> None:
    if final.exists():
        raise FileExistsError(f"recovery checkpoint already exists: {final}")
    os.replace(temporary, final)
    _fsync_directory(final.parent)
    atomic_json(final.parent / "latest.json", latest_payload)


def _release_parameter_names(module: Any) -> set[str]:

    return {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad or name.endswith(_RELEASE_PERSISTENT_PARAMETER_SUFFIXES)
    }


def export_trainable_model_state(
    module: Any, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:

    release_names = _release_parameter_names(module)
    missing = release_names.difference(state_dict)
    if missing:
        raise ValueError(
            "training state is missing release parameters: " + ", ".join(sorted(missing)[:10])
        )
    output: dict[str, torch.Tensor] = {}
    for name in sorted(release_names):
        checkpoint_name = name[len("pipe.") :] if name.startswith("pipe.") else name
        if checkpoint_name in output:
            raise ValueError(f"duplicate recovery parameter name: {checkpoint_name}")
        output[checkpoint_name] = state_dict[name]
    return output


def restore_trainable_model_state(module: Any, state_dict: dict[str, torch.Tensor]) -> None:

    release_names = _release_parameter_names(module)
    expected: dict[str, torch.nn.Parameter] = {}
    for name, parameter in module.named_parameters():
        if name not in release_names:
            continue
        checkpoint_name = name[len("pipe.") :] if name.startswith("pipe.") else name
        if checkpoint_name in expected:
            raise ValueError(f"duplicate trainable parameter name: {checkpoint_name}")
        expected[checkpoint_name] = parameter
    if set(state_dict) != set(expected):
        missing = sorted(set(expected).difference(state_dict))
        extra = sorted(set(state_dict).difference(expected))
        raise ValueError(
            f"recovery model inventory mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            value = state_dict[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"recovery parameter shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            if value.dtype != parameter.dtype:
                raise ValueError(
                    f"recovery parameter dtype mismatch for {name}: {value.dtype} != {parameter.dtype}"
                )
            parameter.copy_(value.to(device=parameter.device, non_blocking=False))


def validate_inventory(root: Path, name: str, expected: dict[str, Any]) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe recovery file name: {name!r}")
    path = (root / candidate).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"recovery file is missing: {path}")
    size = path.stat().st_size
    if size != int(expected.get("size", -1)):
        raise ValueError(f"recovery file size mismatch for {path}: {size}")
    return path


def capture_rank_state(
    module: Any,
    *,
    rank: int,
    update: int,
    microbatches_per_optimizer_update: int = 1,
    cuda_device: torch.device | str | int | None = None,
) -> dict[str, Any]:
    if microbatches_per_optimizer_update <= 0:
        raise ValueError("microbatches_per_optimizer_update must be positive")
    attention = module.attention_runtime
    cuda_rng_state: torch.Tensor | None = None
    cuda_device_index: int | None = None
    if torch.cuda.is_available():
        selected = (
            torch.device("cuda", torch.cuda.current_device())
            if cuda_device is None
            else torch.device(cuda_device)
        )
        if selected.type != "cuda":
            raise ValueError(f"CUDA RNG requested for a non-CUDA device: {selected}")
        cuda_device_index = (
            torch.cuda.current_device() if selected.index is None else int(selected.index)
        )
        cuda_rng_state = torch.cuda.get_rng_state(cuda_device_index)
    return {
        "format": RECOVERY_FORMAT,
        "rank": int(rank),
        "update": int(update),
        "microbatches_per_optimizer_update": int(microbatches_per_optimizer_update),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": cuda_rng_state,
        "torch_cuda_rng_device_index": cuda_device_index,
        "prompt_rng_state": module._prompt_rng.getstate(),
        "prompt_seen": int(module.prompt_seen),
        "prompt_dropped": int(module.prompt_dropped),
        "expert_updates": {str(name): int(value) for name, value in module.expert_updates.items()},
        "attention_counts": {
            "calls": int(attention.calls),
            "plain_calls": int(attention.plain_calls),
            "key_bias_calls": int(attention.key_bias_calls),
        },
    }


def restore_module_progress(
    module: Any,
    state: dict[str, Any],
    *,
    rank: int,
    update: int,
    microbatches_per_optimizer_update: int = 1,
) -> None:
    if state.get("format") != RECOVERY_FORMAT:
        raise ValueError(f"unsupported rank recovery format: {state.get('format')!r}")
    if int(state.get("rank", -1)) != rank or int(state.get("update", -1)) != update:
        raise ValueError("rank recovery identity mismatch")
    if microbatches_per_optimizer_update <= 0:
        raise ValueError("microbatches_per_optimizer_update must be positive")
    recorded_microbatches = int(state.get("microbatches_per_optimizer_update", 1))
    if recorded_microbatches != microbatches_per_optimizer_update:
        raise ValueError(
            f"rank recovery accumulation specification mismatch: {recorded_microbatches} != {microbatches_per_optimizer_update}"
        )
    completed_microbatches = update * microbatches_per_optimizer_update
    expected_experts = set(module.expert_updates)
    restored_experts = {
        str(name): int(value) for name, value in dict(state.get("expert_updates", {})).items()
    }
    if set(restored_experts) != expected_experts:
        raise ValueError("rank recovery expert inventory mismatch")
    if sum(restored_experts.values()) != completed_microbatches:
        raise ValueError("rank recovery expert counts do not match microbatch cursor")
    module.expert_updates = restored_experts
    module.prompt_seen = int(state["prompt_seen"])
    module.prompt_dropped = int(state["prompt_dropped"])
    if (
        module.prompt_seen != completed_microbatches
        or not 0 <= module.prompt_dropped <= module.prompt_seen
    ):
        raise ValueError("rank recovery prompt accounting is inconsistent")
    counts = dict(state["attention_counts"])
    module.attention_runtime.calls = int(counts["calls"])
    module.attention_runtime.plain_calls = int(counts["plain_calls"])
    module.attention_runtime.key_bias_calls = int(counts["key_bias_calls"])


def restore_rng_state(
    module: Any, state: dict[str, Any], *, cuda_device: torch.device | str | int | None = None
) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_state = state["torch_cuda_rng_state"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG recovery requested without CUDA")
        selected = (
            torch.device("cuda", torch.cuda.current_device())
            if cuda_device is None
            else torch.device(cuda_device)
        )
        if selected.type != "cuda":
            raise ValueError(f"CUDA RNG restored onto a non-CUDA device: {selected}")
        selected_index = (
            torch.cuda.current_device() if selected.index is None else int(selected.index)
        )
        torch.cuda.set_rng_state(cuda_state, device=selected_index)
    module._prompt_rng.setstate(state["prompt_rng_state"])
