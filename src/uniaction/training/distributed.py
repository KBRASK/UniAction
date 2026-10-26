from __future__ import annotations
import os
import time
from contextlib import nullcontext
from typing import Any
import accelerate
from accelerate.utils import DistributedDataParallelKwargs
import torch
from torch import nn
from uniaction.backbones.wan.loader import LocalWeightBroadcast

NO_WEIGHT_DECAY_NAME_FRAGMENTS = ("log_reference_count", "token_norm.weight")


class _DDPInitSyncDisabled:
    def __init__(self, delegate: DistributedDataParallelKwargs) -> None:
        self._delegate = delegate

    def to_kwargs(self) -> dict[str, Any]:
        values = self._delegate.to_kwargs()
        values["init_sync"] = False
        return values

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _build_training_accelerator(
    ddp_handler: DistributedDataParallelKwargs,
) -> accelerate.Accelerator:
    return accelerate.Accelerator(
        gradient_accumulation_steps=1,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_handler],
    )


def _configure_dynamo_ddp_optimizer(
    *, world_size: int, regional_compile_mode: str, gradient_checkpointing: bool
) -> dict[str, Any]:

    import torch._dynamo

    requested = torch._dynamo.config.optimize_ddp
    disabled = bool(world_size > 1 and regional_compile_mode != "off" and gradient_checkpointing)
    if disabled:
        torch._dynamo.config.optimize_ddp = False
    return {
        "requested": requested,
        "effective": torch._dynamo.config.optimize_ddp,
        "disabled_for_regional_checkpoint_ddp": disabled,
    }


def _build_local_weight_broadcast(
    accelerator: accelerate.Accelerator, *, enabled: bool
) -> LocalWeightBroadcast | None:
    if not enabled or accelerator.num_processes == 1:
        return None
    if not torch.distributed.is_initialized():
        raise RuntimeError("local weight broadcast requires an initialized process group")
    raw_size = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("UNIACTION_GPUS_PER_NODE")
    if raw_size is None:
        raise RuntimeError(
            "local weight broadcast requires LOCAL_WORLD_SIZE or UNIACTION_GPUS_PER_NODE"
        )
    local_world_size = int(raw_size)
    if local_world_size <= 1 or accelerator.num_processes % local_world_size:
        raise RuntimeError(
            f"invalid local weight broadcast size {local_world_size} for world size {accelerator.num_processes}"
        )
    expected_local_rank = accelerator.process_index % local_world_size
    if accelerator.local_process_index != expected_local_rank:
        raise RuntimeError(
            f"non-contiguous rank layout: global={accelerator.process_index} local={accelerator.local_process_index}, expected={expected_local_rank}"
        )
    group, _all_groups = torch.distributed.new_subgroups(
        group_size=local_world_size, group_desc="uniaction-weight-broadcast"
    )
    source_rank = accelerator.process_index - accelerator.local_process_index
    return LocalWeightBroadcast(
        group=group,
        source_rank=source_rank,
        is_source=accelerator.local_process_index == 0,
        group_size=local_world_size,
    )


def _synchronize_small_module_state(
    module: nn.Module, accelerator: accelerate.Accelerator, *, enabled: bool
) -> dict[str, Any]:

    trainable = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    buffers = list(module.named_buffers())
    parameter_bytes = sum(
        (parameter.numel() * parameter.element_size() for _name, parameter in trainable)
    )
    buffer_bytes = sum((buffer.numel() * buffer.element_size() for _name, buffer in buffers))
    report = {
        "enabled": bool(enabled and accelerator.num_processes > 1),
        "source_rank": 0,
        "collective_device": str(torch.device(accelerator.device)),
        "trainable_tensors": len(trainable),
        "trainable_bytes": int(parameter_bytes),
        "buffer_tensors": len(buffers),
        "buffer_bytes": int(buffer_bytes),
        "seconds": 0.0,
    }
    if not report["enabled"]:
        return report
    if not torch.distributed.is_initialized():
        raise RuntimeError("small-state synchronization requires distributed setup")
    collective_device = torch.device(accelerator.device)
    mismatched_devices = [
        f"{kind}:{name}={tensor.device}"
        for kind, values in (("parameter", trainable), ("buffer", buffers))
        for name, tensor in values
        if tensor.device != collective_device
    ]
    if mismatched_devices:
        raise RuntimeError(
            f"small-state synchronization requires every tensor on the process collective device {collective_device}; mismatches="
            + ", ".join(mismatched_devices[:20])
        )
    started = time.perf_counter()
    with torch.no_grad():
        for _name, parameter in trainable:
            torch.distributed.broadcast(parameter.detach(), src=0)
        for _name, buffer in buffers:
            torch.distributed.broadcast(buffer.detach(), src=0)
    report["seconds"] = time.perf_counter() - started
    return report


def scale_microbatch_loss(
    loss: torch.Tensor, *, microbatches_per_optimizer_update: int
) -> torch.Tensor:

    if microbatches_per_optimizer_update <= 0:
        raise ValueError("microbatches_per_optimizer_update must be positive")
    return loss / float(microbatches_per_optimizer_update)


def gradient_sync_context(
    module: nn.Module,
    *,
    microbatch_index: int,
    microbatches_per_optimizer_update: int,
    world_size: int,
):

    if not 0 <= microbatch_index < microbatches_per_optimizer_update:
        raise ValueError("microbatch_index is outside the accumulation window")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if world_size == 1 or microbatch_index + 1 == microbatches_per_optimizer_update:
        return nullcontext()
    no_sync = getattr(module, "no_sync", None)
    if no_sync is None:
        raise RuntimeError("distributed gradient accumulation requires DDP no_sync")
    return no_sync()


def weight_decay_param_groups(module: Any, weight_decay: float) -> tuple[Any, dict[str, Any]]:

    trainable = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if weight_decay == 0.0:
        return (
            [parameter for _, parameter in trainable],
            {"policy": "uniform", "reason": "weight_decay is zero", "excluded": []},
        )
    decayed: list[Any] = []
    excluded: list[Any] = []
    excluded_names: list[str] = []
    for name, parameter in trainable:
        if any((fragment in name for fragment in NO_WEIGHT_DECAY_NAME_FRAGMENTS)):
            excluded.append(parameter)
            excluded_names.append(name)
        else:
            decayed.append(parameter)
    if len(decayed) + len(excluded) != len(trainable):
        raise RuntimeError("weight-decay grouping lost a trainable parameter")
    identities = {id(p) for p in decayed} | {id(p) for p in excluded}
    if len(identities) != len(trainable):
        raise RuntimeError("weight-decay grouping duplicated a trainable parameter")
    specification = {
        "policy": "exclude_scale_and_norm",
        "fragments": list(NO_WEIGHT_DECAY_NAME_FRAGMENTS),
        "excluded": sorted(excluded_names),
        "decayed_count": len(decayed),
        "excluded_count": len(excluded),
    }
    groups: list[Any] = [{"params": decayed}]
    if excluded:
        groups.append({"params": excluded, "weight_decay": 0.0})
    return (groups, specification)
