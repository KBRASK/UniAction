import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated UniAction videos")
    commands = parser.add_subparsers(dest="metric", required=True)
    similarity = commands.add_parser("similarity", help="LPIPS similarity to the ground truth")
    similarity.add_argument("--video", type=Path, required=True)
    similarity.add_argument("--reference", type=Path, required=True)
    similarity.add_argument("--device", default="cuda")
    force = commands.add_parser("force", help="Optical-flow force response")
    force.add_argument("--video", type=Path, required=True)
    force.add_argument("--condition", type=Path, required=True)
    force.add_argument("--kind", choices=("ball", "domino", "flower"), required=True)
    force.add_argument("--reference-image", type=Path)
    force.add_argument("--calibration", type=Path)
    pose = commands.add_parser("pose", help="CoTracker3 trajectory following")
    inputs = pose.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--tracks", type=Path)
    pose.add_argument("--condition", type=Path, required=True)
    pose.add_argument("--weights", type=Path)
    pose.add_argument("--static", action="store_true")
    pose.add_argument("--device", default="cuda")
    cam = commands.add_parser(
        "camera", help="Continuous direction score from ViPE camera-to-world poses"
    )
    cam.add_argument("--pose", type=Path, required=True)
    cam.add_argument("--null-pose", type=Path, required=True)
    cam.add_argument("--key", choices=tuple("WASDIJKL"), required=True)
    for command in (similarity, force, pose, cam):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from uniaction.evaluation.api import (
            ReconstructionEvaluator,
            ForceEvaluator,
            camera,
            trajectory,
        )
    except ModuleNotFoundError as error:
        parser.error(f"Missing dependency {error.name!r}; install uniaction[eval]")
    if args.output.exists():
        parser.error("Output already exists; choose a new output path")
    track_output = args.output.with_suffix(".npz")
    if args.metric == "pose" and args.video and track_output.exists():
        parser.error("Tracking output already exists; choose a new output path")
    paths = {
        name: value
        for name, value in vars(args).items()
        if isinstance(value, Path) and name != "output" and value is not None
    }
    identities = {name: dict(path=str(path)) for name, path in paths.items()}
    arrays = None
    if args.metric == "similarity":
        result = ReconstructionEvaluator(args.device)(args.video, args.reference)
    elif args.metric == "force":
        with np.load(args.condition, allow_pickle=False) as data:
            if str(data["task"].item()) != "force":
                parser.error("Expected a force condition")
            reference = (
                None
                if args.reference_image is None
                else np.asarray(Image.open(args.reference_image).convert("RGB"))
            )
            result = ForceEvaluator(args.calibration)(
                args.video,
                start=data["start"],
                end=data["end"],
                kind=args.kind,
                reference_image=reference,
            )
    elif args.metric == "pose":
        with np.load(args.condition, allow_pickle=False) as data:
            task = str(data["task"].item())
            if task not in ("hand", "robot"):
                parser.error("Expected a hand or robot condition")
            uv = data["uv"].copy()
            valid = data["visible"].copy() if "visible" in data else np.isfinite(uv).all(-1)
        if args.tracks:
            with np.load(args.tracks, allow_pickle=False) as data:
                result = trajectory(
                    data["measured_px"],
                    data["measured_valid"],
                    uv,
                    valid,
                    task=task,
                    static=args.static,
                )
        else:
            from uniaction.evaluation.pose import PoseEvaluator

            result, arrays = PoseEvaluator(args.weights, args.device)(
                args.video, uv, valid, task=task, static=args.static
            )
    else:
        result = camera(args.pose, args.null_pose, args.key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        with track_output.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        identities["tracks"] = dict(path=str(track_output))
    parameters = {
        name: value
        for name, value in vars(args).items()
        if not isinstance(value, Path) and value is not None
    }
    receipt = dict(
        schema="uniaction-evaluation-v1",
        metric=args.metric,
        parameters=parameters,
        inputs=identities,
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__),
        result=result,
    )
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
