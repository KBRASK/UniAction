import argparse
import json
from pathlib import Path
import shutil

from uniaction.checkpoints import release_metadata


def export(stage1_run, stage2_run, output):
    runs = [Path(stage1_run), Path(stage2_run)]
    paths = [r / f"stage{i}.safetensors" for i, r in enumerate(runs, 1)]
    identities = [json.loads((r / "identity.json").read_text()) for r in runs]
    for stage, run in enumerate(runs, 1):
        if (
            identities[stage - 1]["stage"] != stage
            or json.loads((run / "status.json").read_text())["status"] != "complete"
        ):
            raise ValueError(f"Stage {stage} run is not complete")
    if Path(identities[1]["stage1_checkpoint"]).resolve() != paths[0].resolve():
        raise ValueError("Stage 2 was trained from a different Stage 1 checkpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = release_metadata()
    metadata["schema"] = "uniaction-trained-a14b-v1"
    for stage, path in zip(("stage1", "stage2"), paths):
        target = output / f"{stage}.safetensors"
        shutil.copyfile(path, target)
    from uniaction.training.recovery import atomic_json

    atomic_json(output / "release.json", metadata)
    fields = (
        "stage",
        "seed",
        "optimizer_updates",
        "global_batch_size",
        "prompt_dropout",
        "learning_rate",
        "warmup_updates",
        "weight_decay",
        "reference_count",
        "lora_rank",
    )
    atomic_json(
        output / "training.json",
        {
            f"stage{stage}": {key: identity[key] for key in fields if key in identity}
            for stage, identity in enumerate(identities, 1)
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Package the completed two-stage training outputs for inference."
    )
    parser.add_argument("--stage1-run", required=True)
    parser.add_argument("--stage2-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args.stage1_run, args.stage2_run, args.output))


if __name__ == "__main__":
    main()
