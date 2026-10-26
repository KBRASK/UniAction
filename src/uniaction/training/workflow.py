from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def completed_checkpoint(run, stage):
    from .recovery import validate_inventory

    status_path = run / "status.json"
    if not status_path.exists() or read(status_path)["status"] != "complete":
        return None
    identity = read(run / "identity.json")
    latest = read(run / "recovery/latest.json")
    directory = latest["directory"]
    if Path(directory).name != directory or not directory.startswith("update-"):
        raise ValueError("Invalid recovery directory")
    recovery = run / "recovery" / directory
    manifest = read(validate_inventory(recovery, "manifest.json", latest["manifest"]))
    if (
        identity["stage"] != stage
        or manifest["identity"] != identity
        or manifest["update"] != identity["optimizer_updates"]
        or read(status_path)["optimizer_updates"] != manifest["update"]
    ):
        raise ValueError(f"Incomplete or inconsistent training output: {run}")
    return validate_inventory(
        run, f"stage{stage}.safetensors", manifest["files"]["model.safetensors"]
    )


def training_command(args, root, stage, gpus, resume):
    from .data import training_input

    input_name, input_path = training_input(args)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={gpus}",
        "-m",
        "uniaction.cli.train",
        "--base-model",
        str(args.base_model),
        "--" + input_name.replace("_", "-"),
        str(input_path),
        "--stage",
        str(stage),
        "--output",
        str(root / f"stage{stage}"),
        "--stop-file",
        str(root / ".stop-requested"),
        "--workers",
        str(args.workers),
        "--save-every",
        str(args.save_every),
    ]
    if stage == 2:
        command += ["--stage1-checkpoint", str(root / "stage1/stage1.safetensors")]
    for name in ("bf16_manifest", "cache_dir", "max_updates"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if resume:
        command.append("--resume")
    return command


def train_all(args):
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        raise ValueError(
            "Run uniaction-train directly; it launches distributed workers automatically"
        )
    import torch
    from .recovery import atomic_json
    from .data import training_input, source_specs

    input_name, input_path = training_input(args)
    if input_name == "sources":
        source_specs(input_path)

    gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()
    if gpus not in (1, 2, 4, 8, 16):
        raise ValueError("Training requires 1, 2, 4, 8 or 16 visible GPUs")
    if args.max_updates is not None and not 1 <= args.max_updates <= 2500:
        raise ValueError("max-updates must be in [1, 2500]")
    if args.workers < 0 or args.save_every < 1:
        raise ValueError("workers must be nonnegative and save-every positive")
    identity = dict(
        base_model=str(Path(args.base_model).resolve()),
        training_input=input_name,
        training_input_path=str(Path(input_path).resolve()),
        gpus=gpus,
        optimizer_updates=2500 if args.max_updates is None else args.max_updates,
        bf16_manifest=str(Path(args.bf16_manifest).resolve()) if args.bf16_manifest else None,
    )
    root = Path(args.output).resolve()
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    with (root / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if read(root / "workflow.json") != identity:
                raise ValueError("Resume requires the same data paths, recipe and GPU count")
        else:
            atomic_json(root / "workflow.json", identity)
        stop_file = root / ".stop-requested"
        stop_file.unlink(missing_ok=True)
        stopped = False

        def request_stop(signum, frame):
            nonlocal stopped
            stopped = True
            stop_file.touch()

        handlers = {
            sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            for stage in (1, 2):
                if stopped:
                    return
                run = root / f"stage{stage}"
                if args.resume and completed_checkpoint(run, stage) is not None:
                    continue
                resume = args.resume and run.exists()
                if resume and not (run / "recovery/latest.json").exists():
                    raise RuntimeError(
                        f"No recoverable checkpoint in {run}; use a new output directory"
                    )
                subprocess.run(
                    training_command(args, root, stage, gpus, resume),
                    check=True,
                    start_new_session=True,
                )
                status = read(run / "status.json")
                if stopped or status.get("stop_requested") or status["status"] == "paused":
                    print(
                        f"Training paused. Repeat the same command with --resume: {root}",
                        flush=True,
                    )
                    return
                if completed_checkpoint(run, stage) is None:
                    raise RuntimeError(f"Training did not finish successfully: {run}")
            if stopped:
                return
            from uniaction.cli.export import export
            from uniaction.checkpoints import verify_checkpoints

            output = root / "checkpoint"
            if output.exists():
                verify_checkpoints(
                    output / "stage1.safetensors", output / "stage2.safetensors", trained=True
                )
            else:
                temporary = root / (".checkpoint-" + uuid.uuid4().hex)
                export(root / "stage1", root / "stage2", temporary)
                temporary.rename(output)
            print(f"Training complete. Checkpoint: {output}", flush=True)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
