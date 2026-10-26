from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import uuid

import torch
from accelerate.utils import (
    DistributedDataParallelKwargs,
    broadcast_object_list,
    gather_object,
    set_seed,
)
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from .config import make_training_config
from .data import load_dataset, load_sources, single_example, training_input
from .distributed import (
    _build_training_accelerator,
    _DDPInitSyncDisabled,
    _build_local_weight_broadcast,
    _configure_dynamo_ddp_optimizer,
    _synchronize_small_module_state,
    gradient_sync_context,
    scale_microbatch_loss,
    weight_decay_param_groups,
)
from .model import UniActionTrainingModule
from .recovery import (
    atomic_json,
    atomic_safetensors,
    atomic_torch_save,
    capture_rank_state,
    commit_recovery_directory,
    export_trainable_model_state,
    inventory_file,
    restore_module_progress,
    restore_rng_state,
    restore_trainable_model_state,
    validate_inventory,
)


def main_call(accelerator, function):

    result = [None]
    if accelerator.is_main_process:
        try:
            result[0] = {"ok": True, "value": function()}
        except Exception as error:
            result[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    broadcast_object_list(result)
    if not result[0]["ok"]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]


def save_checkpoint(accelerator, module, optimizer, scheduler, run, update, accumulation, identity):
    unwrapped = accelerator.unwrap_model(module)
    temporary_name = main_call(accelerator, lambda: ".pending-" + uuid.uuid4().hex)
    root = run / "recovery"
    temporary = root / temporary_name
    main_call(accelerator, lambda: temporary.mkdir(parents=True, exist_ok=False))
    rank_state = capture_rank_state(
        unwrapped,
        rank=accelerator.process_index,
        update=update,
        microbatches_per_optimizer_update=accumulation,
        cuda_device=accelerator.device,
    )
    errors = []
    try:
        atomic_torch_save(temporary / f"rank-{accelerator.process_index:05d}.pt", rank_state)
    except Exception as error:
        errors.append(f"Rank {accelerator.process_index}: {type(error).__name__}: {error}")
    errors = gather_object(errors)
    if errors:
        raise RuntimeError("Recovery write failed: " + "; ".join(errors))

    def commit():
        model = export_trainable_model_state(unwrapped, unwrapped.state_dict())
        atomic_safetensors(temporary / "model.safetensors", model)
        atomic_torch_save(
            temporary / "optimizer.pt",
            {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        )
        files = {p.name: inventory_file(p) for p in temporary.iterdir() if p.is_file()}
        final = root / f"update-{update:08d}"
        manifest = dict(
            schema="uniaction-training-transaction-v1",
            update=update,
            world_size=accelerator.num_processes,
            accumulation=accumulation,
            identity=identity,
            files=files,
        )
        info = atomic_json(temporary / "manifest.json", manifest)
        commit_recovery_directory(
            temporary, final, latest_payload={"directory": final.name, "manifest": info}
        )
        return str(final)

    return main_call(accelerator, commit)


def load_checkpoint(accelerator, module, optimizer, scheduler, run, accumulation, identity):
    def resolve():
        latest = json.loads((run / "recovery/latest.json").read_text())
        name = latest["directory"]
        if Path(name).name != name or not name.startswith("update-"):
            raise ValueError("Invalid recovery directory")
        root = run / "recovery" / name
        manifest_path = validate_inventory(root, "manifest.json", latest["manifest"])
        manifest = json.loads(manifest_path.read_text())
        if (manifest["identity"], manifest["world_size"], manifest["accumulation"]) != (
            identity,
            accelerator.num_processes,
            accumulation,
        ):
            raise ValueError("Resume requires the same data, recipe, stage and process count")
        for filename, expected in manifest["files"].items():
            validate_inventory(root, filename, expected)
        return dict(root=str(root), update=manifest["update"])

    restored = main_call(accelerator, resolve)
    root, update = Path(restored["root"]), int(restored["update"])
    unwrapped = accelerator.unwrap_model(module)
    restore_trainable_model_state(unwrapped, load_file(str(root / "model.safetensors")))

    state = torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rank = torch.load(
        root / f"rank-{accelerator.process_index:05d}.pt", map_location="cpu", weights_only=False
    )
    restore_module_progress(
        unwrapped,
        rank,
        rank=accelerator.process_index,
        update=update,
        microbatches_per_optimizer_update=accumulation,
    )
    optimizer.zero_grad()
    return update, rank


def finish_run(run, stage, checkpoint, update, total_updates, stopped):
    if update == total_updates:
        source = Path(checkpoint) / "model.safetensors"
        target = run / f"stage{stage}.safetensors"
        if target.exists():
            if source.stat().st_size != target.stat().st_size:
                raise ValueError("Existing final checkpoint differs from recovery state")
        else:
            os.link(source, target)
    atomic_json(
        run / "status.json",
        dict(
            status="complete" if update == total_updates else "paused",
            stop_requested=stopped,
            optimizer_updates=update,
            sample_exposures=16 * update,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )


def train(args):

    handler = DistributedDataParallelKwargs(
        find_unused_parameters=args.stage == 2,
        static_graph=False,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
    accelerator = _build_training_accelerator(handler)
    accelerator.ddp_handler = _DDPInitSyncDisabled(accelerator.ddp_handler)
    world = accelerator.num_processes
    if accelerator.device.type != "cuda":
        raise RuntimeError("A14B training requires CUDA")
    if world > 16 or 16 % world:
        raise ValueError("Use 1, 2, 4, 8 or 16 processes; effective batch size stays 16")
    accumulation = 16 // world
    input_name, input_path = training_input(args)
    encoder = None
    if input_name == "sources":
        from .online import OnlineEncoder

        dataset = load_sources(input_path)
        encoder = OnlineEncoder(args.base_model, accelerator.device)
        empty_context = None
    else:
        dataset, empty_context, _ = load_dataset(input_path)
    config = make_training_config(
        args.base_model,
        args.stage,
        empty_context=empty_context,
        stage1_checkpoint=args.stage1_checkpoint,
        bf16_manifest=args.bf16_manifest,
    )
    from uniaction.model.wan import validate_wan_training_assets

    def identify_inputs():
        assets = validate_wan_training_assets(config)

        weights = assets["verified_mirrors"] or [
            dict(name=Path(p).name, **inventory_file(Path(p))) for p in assets["expert_files"]
        ]
        return dict(
            weights=weights,
            stage1_checkpoint=str(Path(args.stage1_checkpoint).resolve())
            if args.stage == 2
            else None,
        )

    inputs = main_call(accelerator, identify_inputs)
    total_updates = 2500 if args.max_updates is None else args.max_updates
    if not 1 <= total_updates <= 2500:
        raise ValueError("max-updates must be in [1, 2500]")
    if args.save_every < 1 or args.workers < 0:
        raise ValueError("save-every must be positive and workers nonnegative")
    from uniaction.runtime_profiles import resolve_runtime_profile

    profile = resolve_runtime_profile(
        stage=args.stage,
        device_name=torch.cuda.get_device_name(accelerator.device),
        compute_capability=torch.cuda.get_device_capability(accelerator.device),
        model_family=config.model.family,
        num_frames=config.model.num_frames,
    )
    compile_mode = profile.regional_compile_mode
    _configure_dynamo_ddp_optimizer(
        world_size=world, regional_compile_mode=compile_mode, gradient_checkpointing=True
    )
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    broadcast = _build_local_weight_broadcast(accelerator, enabled=local_size > 1)
    module = UniActionTrainingModule(
        config,
        args.stage,
        **profile.training_kwargs(),
        load_device=accelerator.device,
        local_weight_broadcast=broadcast,
        cache_dir=args.cache_dir,
        empty_prompt_context=encoder.empty_context if encoder is not None else None,
    )
    module.to(accelerator.device)
    module.resolve_attention_backend(accelerator.device)
    _synchronize_small_module_state(module, accelerator, enabled=True)
    set_seed(config.seed, device_specific=True)
    groups, _ = weight_decay_param_groups(module, config.optimization.weight_decay)
    optimizer = torch.optim.AdamW(
        groups,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda update: min((update + 1) / 100, 1.0)
    )
    loader_args = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=single_example,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    if args.workers:
        loader_args["prefetch_factor"] = 2
        loader_args["multiprocessing_context"] = "spawn"
    loader = DataLoader(**loader_args)
    module, optimizer, loader, scheduler = accelerator.prepare(module, optimizer, loader, scheduler)
    run = Path(args.output).resolve()
    identity = dict(
        schema="uniaction-training-v1",
        stage=args.stage,
        seed=config.seed,
        training_input=input_name,
        training_input_path=str(Path(input_path).resolve()),
        base_model=str(Path(args.base_model).resolve()),
        stage1_checkpoint=inputs["stage1_checkpoint"],
        inputs=inputs,
        bf16_manifest=str(config.model.dit_bf16_manifest)
        if config.model.dit_bf16_manifest
        else None,
        optimizer_updates=total_updates,
        global_batch_size=16,
        world_size=world,
        prompt_dropout=0.1 if args.stage == 1 else 0.0,
        learning_rate=7.5e-5,
        warmup_updates=100,
        weight_decay=0.01,
        reference_count=784,
        lora_rank=128,
        gradient_checkpointing=True,
        compile_mode=compile_mode,
    )

    def prepare_run():
        if args.resume:
            if json.loads((run / "identity.json").read_text()) != identity:
                raise ValueError("Training identity changed")
        else:
            run.mkdir(parents=True, exist_ok=False)
            atomic_json(run / "identity.json", identity)

    main_call(accelerator, prepare_run)
    completed, rank_state = (0, None)
    if args.resume:
        completed, rank_state = load_checkpoint(
            accelerator, module, optimizer, scheduler, run, accumulation, identity
        )
    if completed == total_updates:

        def finish_recovered():
            latest = json.loads((run / "recovery/latest.json").read_text())
            finish_run(
                run,
                args.stage,
                run / "recovery" / latest["directory"],
                completed,
                total_updates,
                False,
            )

        main_call(accelerator, finish_recovered)
        accelerator.print("This run is already complete.")
        accelerator.end_training()
        return
    loader = accelerator.skip_first_batches(loader, completed * accumulation)
    iterator = iter(loader)
    unwrapped = accelerator.unwrap_model(module)
    if rank_state is not None:
        restore_rng_state(unwrapped, rank_state, cuda_device=accelerator.device)
    requested = {"stop": False}
    old_handlers = {}

    def request_stop(signum, frame):
        requested["stop"] = True

    for sig in (signal.SIGTERM,):
        old_handlers[sig] = signal.signal(sig, request_stop)
    optimizer.zero_grad()
    stopped = False
    try:
        for update in range(completed + 1, total_updates + 1):
            loss_sum = torch.zeros((), device=accelerator.device)
            for micro in range(accumulation):
                sample = next(iterator)
                if encoder is not None:
                    sample = encoder(sample)
                with gradient_sync_context(
                    module,
                    microbatch_index=micro,
                    microbatches_per_optimizer_update=accumulation,
                    world_size=world,
                ):
                    try:
                        loss = module(sample)
                        accelerator.backward(
                            scale_microbatch_loss(
                                loss, microbatches_per_optimizer_update=accumulation
                            )
                        )
                        loss_sum += loss.detach() / accumulation
                    finally:
                        unwrapped.clear_staged_action()
            accelerator.clip_grad_norm_(unwrapped.trainable_parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            stop_file = getattr(args, "stop_file", None)
            stop = torch.tensor(
                int(requested["stop"] or (stop_file and Path(stop_file).exists()) or False),
                device=accelerator.device,
            )
            if world > 1:
                torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
            stopped = bool(stop.item())
            mean_loss = accelerator.reduce(loss_sum, reduction="mean").item()
            if accelerator.is_main_process:
                with (run / "loss.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            dict(
                                update=update,
                                sample_exposures=16 * update,
                                loss=mean_loss,
                                learning_rate=optimizer.param_groups[0]["lr"],
                            )
                        )
                        + "\n"
                    )
            if update % args.save_every == 0 or stopped or update == total_updates:
                checkpoint = save_checkpoint(
                    accelerator, module, optimizer, scheduler, run, update, accumulation, identity
                )
                accelerator.print(
                    f"Update {update}/{total_updates}: loss={mean_loss:.6f}; {checkpoint}"
                )
            if stopped:
                break

        main_call(
            accelerator,
            lambda: finish_run(run, args.stage, checkpoint, update, total_updates, stopped),
        )
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
        accelerator.end_training()
