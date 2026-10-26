from __future__ import annotations

import json
from pathlib import Path


def release_metadata():
    return {
        "schema": "uniaction-a14b-v1",
        "model_family": "Wan2.2-I2V-A14B-HighNoise",
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "action_rope_mode": "cell_center",
        "action_reads_video": True,
        "reference_count": 784,
        "lora_rank": 128,
        "lora_alpha": 128,
        "stage1": {"checkpoint": "stage1.safetensors"},
        "stage2": {"checkpoint": "stage2.safetensors"},
    }


def read_release(checkpoint_dir, *, trained=False):
    root = Path(checkpoint_dir)
    with (root / "release.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = release_metadata()
    if trained:
        expected["schema"] = "uniaction-trained-a14b-v1"
    if any(
        metadata.get(key) != value
        for key, value in expected.items()
        if key not in ("stage1", "stage2")
    ) or any(
        metadata.get(stage, {}).get("checkpoint") != expected[stage]["checkpoint"]
        for stage in ("stage1", "stage2")
    ):
        raise ValueError("release.json does not describe the released UniAction model")
    return metadata


def verify_checkpoints(stage1, stage2, *, trained=False):
    paths = (Path(stage1).resolve(), Path(stage2).resolve())

    metadata = read_release(Path(stage2).parent, trained=trained)
    for path, stage in zip(paths, ("stage1", "stage2")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {stage} checkpoint: {path}")
    return metadata
