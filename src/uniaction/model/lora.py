from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open


SHARED_TARGETS = ("q", "k", "v", "o", "ffn.0", "ffn.2")
DEFAULT_ADAPTER = "default"


class LoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, *, rank: int, alpha: int) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRA requires nn.Linear, got {type(base_layer).__name__}")
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        self._disable_adapters = False
        self._merged = False
        factory = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        self.lora_A = nn.ModuleDict(
            {
                DEFAULT_ADAPTER: nn.Linear(
                    base_layer.in_features,
                    rank,
                    bias=False,
                    **factory,
                )
            }
        )
        self.lora_B = nn.ModuleDict(
            {
                DEFAULT_ADAPTER: nn.Linear(
                    rank,
                    base_layer.out_features,
                    bias=False,
                    **factory,
                )
            }
        )
        nn.init.zeros_(self.lora_B[DEFAULT_ADAPTER].weight)
        self.base_layer.requires_grad_(False)
        self.lora_A[DEFAULT_ADAPTER].requires_grad_(True)
        self.lora_B[DEFAULT_ADAPTER].requires_grad_(True)

    def get_base_layer(self) -> nn.Linear:
        return self.base_layer

    def _delta_weight(self) -> torch.Tensor:
        a = self.lora_A[DEFAULT_ADAPTER].weight
        b = self.lora_B[DEFAULT_ADAPTER].weight
        if a.device.type == "cpu" and a.dtype in (torch.float16, torch.bfloat16):
            delta = (b.float() @ a.float()).to(dtype=a.dtype)
        else:
            delta = b @ a
        return delta * self.scaling

    def merge(self, safe_merge: bool = True) -> None:
        if self._merged:
            return
        delta = self._delta_weight().to(dtype=self.base_layer.weight.dtype)
        if safe_merge:
            merged = self.base_layer.weight.detach().clone()
            merged += delta
            if not torch.isfinite(merged).all():
                raise ValueError("NaNs detected while merging the LoRA adapter")
            self.base_layer.weight.data = merged
        else:
            self.base_layer.weight.data += delta
        self._merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self._disable_adapters or self._merged:
            return result
        result_dtype = result.dtype
        a = self.lora_A[DEFAULT_ADAPTER]
        b = self.lora_B[DEFAULT_ADAPTER]
        update = b(a(x.to(dtype=a.weight.dtype))) * self.scaling
        return (result + update).to(dtype=result_dtype)


def action_branch_targets(num_blocks: int) -> tuple[str, ...]:
    targets: list[str] = []
    for block in range(num_blocks):
        targets.extend(f"blocks.{block}.self_attn.{name}" for name in ("q", "k", "v", "o"))
        targets.extend((f"blocks.{block}.ffn.0", f"blocks.{block}.ffn.2"))
    return tuple(targets)


def inject_lora(
    model: nn.Module,
    *,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: int,
    dtype: torch.dtype,
) -> nn.Module:
    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    targets = tuple(str(target) for target in target_modules)
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and any(name == target or name.endswith(f".{target}") for target in targets)
    ]
    if not selected:
        raise ValueError(f"no linear modules matched LoRA targets {targets}")
    model.requires_grad_(False)
    for name, layer in selected:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoraLinear(layer, rank=rank, alpha=alpha))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=dtype)
    return model


def stable_reinitialize_lora(model: nn.Module, seed: int) -> int:

    initialized = 0
    for module_name, layer in sorted(model.named_modules()):
        if not isinstance(layer, LoraLinear):
            continue
        for adapter_name, a_layer in sorted(layer.lora_A.items()):
            parameter_seed = (int(seed) + initialized) % (2**63 - 1)

            generator = torch.Generator(device="cpu")
            generator.manual_seed(parameter_seed)
            canonical = torch.empty_like(a_layer.weight, device="cpu")
            nn.init.kaiming_uniform_(
                canonical,
                a=math.sqrt(5),
                generator=generator,
            )
            with torch.no_grad():
                a_layer.weight.copy_(canonical)
            if adapter_name not in layer.lora_B:
                raise RuntimeError(f"{module_name}: LoRA A has no matching B")
            b_layer = layer.lora_B[adapter_name]
            nn.init.zeros_(b_layer.weight)
            if b_layer.bias is not None:
                nn.init.zeros_(b_layer.bias)
            initialized += 1
    if initialized == 0:
        raise RuntimeError("no LoRA layers were found for stable initialization")
    return initialized


def _checkpoint_dit_state(
    path: str | Path,
    *,
    dit: nn.Module,
    prefix: str = "dit",
    local_broadcast=None,
) -> dict[str, torch.Tensor]:
    namespace = f"{prefix}."
    expected = dit.state_dict()
    try:
        model_device = next(dit.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    with safe_open(str(path), framework="pt", device=str(model_device)) as handle:
        external_keys = [key for key in handle.keys() if key.startswith(namespace)]
        if not external_keys:
            raise ValueError(f"checkpoint has no {prefix}.* tensors: {path}")
        internal_keys = [key[len(namespace) :] for key in external_keys]
        unexpected = [key for key in internal_keys if key not in expected]
        if unexpected:
            raise RuntimeError("unexpected checkpoint keys: " + ", ".join(unexpected[:20]))
        mismatched = [
            internal
            for external, internal in zip(external_keys, internal_keys, strict=True)
            if tuple(handle.get_slice(external).get_shape()) != tuple(expected[internal].shape)
        ]
        if mismatched:
            raise RuntimeError("checkpoint tensor shape mismatch: " + ", ".join(mismatched[:20]))
        if local_broadcast is None or local_broadcast.is_source:
            state = {
                internal: handle.get_tensor(external).to(
                    device=model_device,
                    dtype=expected[internal].dtype,
                )
                for external, internal in zip(external_keys, internal_keys, strict=True)
            }
        else:
            state = {
                internal: torch.empty_like(expected[internal], device=model_device)
                for internal in internal_keys
            }
    if local_broadcast is not None:
        if model_device.type != "cuda":
            raise ValueError("local checkpoint broadcast requires CUDA")
        for internal in internal_keys:
            torch.distributed.broadcast(
                state[internal],
                src=local_broadcast.source_rank,
                group=local_broadcast.group,
            )
    return state


def load_dit_checkpoint(
    dit: nn.Module,
    path: str | Path,
    *,
    require_adapter: bool,
    prefix: str = "dit",
    local_broadcast=None,
) -> int:
    state = _checkpoint_dit_state(
        path,
        dit=dit,
        prefix=prefix,
        local_broadcast=local_broadcast,
    )
    has_adapter = any(key.startswith("_action_adapter_ref.") for key in state)
    if has_adapter != require_adapter:
        raise ValueError(
            f"checkpoint adapter presence={has_adapter}, expected={require_adapter}: {path}"
        )
    result = dit.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError("unexpected checkpoint keys: " + ", ".join(result.unexpected_keys[:20]))
    loaded = len(state) - len(result.unexpected_keys)
    if loaded != len(state):
        raise RuntimeError(f"loaded only {loaded}/{len(state)} checkpoint tensors")
    return loaded


def merge_and_unwrap_lora(model: nn.Module) -> int:
    merged = 0
    for name, child in list(model.named_children()):
        if isinstance(child, LoraLinear):
            child.merge(safe_merge=True)
            setattr(model, name, child.get_base_layer())
            merged += 1
        else:
            merged += merge_and_unwrap_lora(child)
    return merged


def load_and_merge_stage1(
    dit: nn.Module,
    checkpoint: str | Path,
    *,
    rank: int = 32,
    alpha: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "dit",
    local_broadcast=None,
) -> nn.Module:
    dit = inject_lora(
        dit,
        target_modules=SHARED_TARGETS,
        rank=rank,
        alpha=alpha,
        dtype=dtype,
    )
    loaded = load_dit_checkpoint(
        dit,
        checkpoint,
        require_adapter=False,
        prefix=prefix,
        local_broadcast=local_broadcast,
    )
    expected_layers = loaded // 2
    merged = merge_and_unwrap_lora(dit)
    if merged != expected_layers:
        raise RuntimeError(f"merged {merged} Stage-1 layers, expected {expected_layers}")
    if any("lora_" in name for name, _ in dit.named_parameters()):
        raise RuntimeError("Stage-1 merge left live LoRA parameters")
    if hasattr(dit, "peft_config"):
        delattr(dit, "peft_config")
    dit.requires_grad_(False)
    return dit


def inject_action_branch_lora(
    dit: nn.Module,
    *,
    rank: int = 32,
    alpha: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    init_seed: int | None = None,
) -> nn.Module:
    dit = inject_lora(
        dit,
        target_modules=action_branch_targets(len(dit.blocks)),
        rank=rank,
        alpha=alpha,
        dtype=dtype,
    )
    if init_seed is not None:
        initialized = stable_reinitialize_lora(dit, init_seed)
        expected = len(dit.blocks) * 6
        if initialized != expected:
            raise RuntimeError(f"initialized {initialized} action LoRAs, expected {expected}")
    dit._action_lora_routing = "action_branch"
    return dit


def trainable_state_dict(module: nn.Module, *, remove_prefix: str = "pipe.") -> dict:
    trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    output = {}
    for name, value in module.state_dict().items():
        if name not in trainable:
            continue
        key = name[len(remove_prefix) :] if name.startswith(remove_prefix) else name
        output[key] = value.detach()
    return output


def lora_layer_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, layer in model.named_modules() if isinstance(layer, LoraLinear))


@contextmanager
def disable_lora_adapters(model: nn.Module):

    layers = [layer for layer in model.modules() if isinstance(layer, LoraLinear)]
    previous = [bool(layer._disable_adapters) for layer in layers]
    for layer in layers:
        layer._disable_adapters = True
    try:
        yield
    finally:
        for layer, disabled in zip(layers, previous, strict=True):
            layer._disable_adapters = disabled
