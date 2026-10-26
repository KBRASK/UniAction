# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for UniAction; upstream copyright and Apache-2.0 attribution are in NOTICE.
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
from safetensors import safe_open

from uniaction.profiles import WanModelProfile

from .dit import WanModel


@dataclass(frozen=True)
class LocalWeightBroadcast:
    group: Any
    source_rank: int
    is_source: bool
    group_size: int


@contextmanager
def _skip_parameter_initialization(device: torch.device = torch.device("meta")):

    original = torch.nn.Module.register_parameter

    def register_empty_parameter(module, name, parameter):
        original(module, name, parameter)
        if parameter is not None:
            parameter_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = parameter.requires_grad
            module._parameters[name] = parameter_cls(module._parameters[name].to(device), **kwargs)

    torch.nn.Module.register_parameter = register_empty_parameter
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = original


class _TensorFiles(Mapping[str, torch.Tensor]):
    def __init__(
        self,
        paths: Iterable[Path],
        *,
        dtype: torch.dtype,
        device: str | torch.device = "cpu",
        key_transform: Callable[[str], str | None] | None = None,
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("checkpoint file list is empty")
        self.dtype = dtype
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        self.device = resolved_device
        self.key_transform = key_transform
        self._safe_files = []
        self._binary: dict[str, torch.Tensor] = {}
        source: dict[str, tuple[str, str, int | None]] = {}
        for path in self.paths:
            if path.suffix == ".safetensors":
                handle = safe_open(str(path), framework="pt", device=str(self.device))
                file_index = len(self._safe_files)
                self._safe_files.append(handle)
                entries = ((key, ("safe", file_index)) for key in handle.keys())
            else:
                loaded = torch.load(path, map_location=self.device, weights_only=True)
                if len(loaded) == 1:
                    for wrapper in ("state_dict", "module", "model_state"):
                        if wrapper in loaded and isinstance(loaded[wrapper], dict):
                            loaded = loaded[wrapper]
                            break
                self._binary.update(loaded)
                entries = ((key, ("binary", None)) for key in loaded)
            for external_name, location in entries:
                internal_name = (
                    key_transform(external_name) if key_transform is not None else external_name
                )
                if internal_name is None:
                    continue
                if internal_name in source:
                    raise ValueError(f"duplicate checkpoint tensor {internal_name!r}")
                kind, index = location
                source[internal_name] = (external_name, kind, index)
        self._source = source

    def __iter__(self):
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, name: str) -> torch.Tensor:
        external, kind, index = self._source[name]
        if kind == "safe":
            assert index is not None
            value = self._safe_files[index].get_tensor(external)
        else:
            value = self._binary[external]
        if isinstance(value, torch.Tensor):
            value = value.to(device=self.device, dtype=self.dtype)

            if value.device.type == "cpu":
                value = value.clone()
        return value

    def shape(self, name: str) -> tuple[int, ...]:
        external, kind, index = self._source[name]
        if kind == "safe":
            assert index is not None
            return tuple(self._safe_files[index].get_slice(external).get_shape())
        return tuple(self._binary[external].shape)

    def storage_dtypes(self) -> tuple[str, ...]:
        values = set()
        for external, kind, index in self._source.values():
            if kind == "safe":
                assert index is not None
                values.add(str(self._safe_files[index].get_slice(external).get_dtype()))
            else:
                values.add(str(self._binary[external].dtype))
        return tuple(sorted(values))


def _vae_key_transform(name: str) -> str:
    return f"model.{name}"


def _load_model(
    model_class,
    paths: Iterable[Path],
    *,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    kwargs: dict | None = None,
    key_transform: Callable[[str], str | None] | None = None,
    local_broadcast: LocalWeightBroadcast | None = None,
):
    started = time.perf_counter()
    with _skip_parameter_initialization():
        model = model_class(**(kwargs or {}))
    constructed = time.perf_counter()
    tensors = _TensorFiles(
        paths,
        dtype=dtype,
        device=device,
        key_transform=key_transform,
    )
    indexed = time.perf_counter()
    expected_state = model.state_dict()
    tensor_names = tuple(tensors)
    if set(tensor_names) != set(expected_state):
        missing = sorted(set(expected_state) - set(tensor_names))
        unexpected = sorted(set(tensor_names) - set(expected_state))
        raise RuntimeError(
            f"checkpoint schema mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    mismatched_shapes = [
        name for name in tensor_names if tensors.shape(name) != tuple(expected_state[name].shape)
    ]
    if mismatched_shapes:
        raise RuntimeError("checkpoint tensor shape mismatch: " + ", ".join(mismatched_shapes[:20]))
    if local_broadcast is None or local_broadcast.is_source:
        state = {name: tensors[name] for name in tensor_names}
        weight_source = "filesystem"
    else:
        state = {
            name: torch.empty(
                tensors.shape(name),
                dtype=dtype,
                device=tensors.device,
            )
            for name in tensor_names
        }
        weight_source = "node_local_broadcast"
    materialized = time.perf_counter()
    broadcast_started = materialized
    if local_broadcast is not None:
        if tensors.device.type != "cuda":
            raise ValueError("local weight broadcast requires CUDA loading")
        for name in tensor_names:
            torch.distributed.broadcast(
                state[name],
                src=local_broadcast.source_rank,
                group=local_broadcast.group,
            )
    broadcast_finished = time.perf_counter()
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    assigned = time.perf_counter()
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint schema mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device=device, dtype=dtype).eval()
    finished = time.perf_counter()
    model._uniaction_load_timing = {
        "device": str(torch.device(device)),
        "meta_construction_seconds": constructed - started,
        "tensor_index_seconds": indexed - constructed,
        "tensor_materialization_seconds": materialized - indexed,
        "node_local_broadcast_seconds": broadcast_finished - broadcast_started,
        "weight_source": weight_source,
        "node_local_broadcast_group_size": (
            1 if local_broadcast is None else local_broadcast.group_size
        ),
        "checkpoint_storage_dtypes": list(tensors.storage_dtypes()),
        "checkpoint_file_bytes": sum(path.stat().st_size for path in tensors.paths),
        "state_assignment_seconds": assigned - broadcast_finished,
        "finalize_seconds": finished - assigned,
        "total_seconds": finished - started,
    }
    return model


def dit_kwargs(profile: WanModelProfile) -> dict:
    common = {
        "patch_size": [1, 2, 2],
        "dim": profile.hidden_dim,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": profile.out_dim,
        "num_heads": profile.num_heads,
        "num_layers": profile.num_blocks,
        "eps": 1e-6,
        "in_dim": profile.in_dim,
    }
    if profile.family in {
        "Wan2.2-I2V-A14B",
        "Wan2.2-I2V-A14B-HighNoise",
    }:
        return {
            **common,
            "has_image_input": False,
            "ffn_dim": 13824,
            "require_clip_embedding": False,
        }
    raise ValueError(f"unsupported Wan profile {profile.family!r}")


def load_dit(
    profile: WanModelProfile,
    paths: Iterable[Path],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    local_broadcast: LocalWeightBroadcast | None = None,
) -> WanModel:
    return _load_model(
        WanModel,
        paths,
        dtype=dtype,
        device=device,
        kwargs=dit_kwargs(profile),
        local_broadcast=local_broadcast,
    )


def load_text_encoder(
    path: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
):

    from .text_encoder import WanTextEncoder

    return _load_model(WanTextEncoder, (path,), dtype=dtype, device=device)


def load_vae(
    profile: WanModelProfile,
    path: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
):
    from .vae import WanVideoVAE

    if profile.family not in {"Wan2.2-I2V-A14B", "Wan2.2-I2V-A14B-HighNoise"}:
        raise ValueError(f"unsupported Wan profile {profile.family!r}")
    return _load_model(
        WanVideoVAE,
        (path,),
        dtype=dtype,
        device=device,
        key_transform=_vae_key_transform,
    )


def load_tokenizer(path: Path):
    from .text_encoder import HuggingfaceTokenizer

    return HuggingfaceTokenizer(name=str(path), seq_len=512, clean="whitespace")
