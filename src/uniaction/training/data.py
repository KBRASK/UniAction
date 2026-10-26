from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import ConcatDataset, Dataset


TASKS = {
    "hand": "hand",
    "force_balls": "force",
    "force_dominos": "force",
    "force_flowers": "force",
    "robot": "robot",
    "camera": "camera",
}
COUNTS = {
    "hand": 10000,
    "force_balls": 5000,
    "force_dominos": 2500,
    "force_flowers": 2500,
    "robot": 10000,
    "camera": 10000,
}


def checked_child(root, name):
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def source_specs(manifest):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    if data.get("schema") != "uniaction-training-sources":
        raise ValueError("Expected training-sources.json from the training set")
    rows = data["streams"]
    if len(rows) != len(TASKS) or {row["name"] for row in rows} != set(TASKS):
        raise ValueError("The training source manifest must contain every stream exactly once")
    entries = {row["name"]: row for row in rows}
    prepared = []
    for name in TASKS:
        row = entries[name]
        metadata = checked_child(manifest.parent, row["metadata"])
        root = checked_child(manifest.parent, row["source_root"])
        if row["samples"] != COUNTS[name]:
            raise ValueError(f"Incorrect source count for {name}")
        if not metadata.is_file():
            raise FileNotFoundError(metadata)
        if not root.is_dir():
            raise FileNotFoundError(f"Missing training media: {root}")
        prepared.append(dict(row, metadata=metadata, source_root=root))
    return prepared


def training_input(args):
    sources = getattr(args, "sources", None)
    cache = getattr(args, "cache_manifest", None)
    if bool(sources) == bool(cache):
        raise ValueError("Provide either --sources or --cache-manifest")
    return ("sources", sources) if sources else ("cache_manifest", cache)


class RawExamples(Dataset):
    def __init__(self, dataset, name):
        self.dataset, self.task = dataset, TASKS[name]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return dict(self.dataset[index], task=self.task)


def load_sources(manifest):
    from .prepare import source_dataset

    return ConcatDataset(
        [
            RawExamples(
                source_dataset(row["name"], row["source_root"], row["metadata"]), row["name"]
            )
            for row in source_specs(manifest)
        ]
    )


class CachedExamples(Dataset):
    def __init__(self, root, *, name):
        self.root = Path(root).resolve()
        self.name, self.task = name, TASKS[name]
        index = json.loads((self.root / "index.json").read_text())
        self.split = index["format"] == "uniaction-split-training-cache"
        if index["format"] not in (
            "uniaction-training-cache",
            "uniaction-split-training-cache",
        ):
            raise ValueError("Training requires an A14B I2V cache")
        config = index["config"]
        if (config["model_family"], config["num_frames"], config["height"], config["width"]) != (
            "Wan2.2-I2V-A14B-HighNoise",
            81,
            480,
            832,
        ):
            raise ValueError("Cached geometry does not match the released training recipe")
        self.records = index["records"]
        self.verified = {}
        self.contexts = OrderedDict()

    def __len__(self):
        return len(self.records)

    def _load(self, row, key):
        path = checked_child(self.root, row[key])
        info = path.stat()
        signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if self.verified.get(str(path)) != signature:
            if row.get(key + "_size", info.st_size) != info.st_size:
                raise ValueError("Cache size mismatch")
            self.verified[str(path)] = signature
        if key == "context" and str(path) in self.contexts:
            self.contexts.move_to_end(str(path))
            return self.contexts[str(path)]
        tensors = load_file(str(path), device="cpu")
        if key == "context":
            self.contexts[str(path)] = tensors
            if len(self.contexts) > 64:
                self.contexts.popitem(last=False)
        return tensors

    def __getitem__(self, index):
        row = self.records[index]
        tensors = self._load(row, "sample")
        if self.split:
            visual = self._load(row, "visual")
            if set(tensors) & set(visual):
                raise ValueError("Cached visual/action tensors overlap")
            tensors.update(visual)
        result = dict(row.get("fields", {}))
        result.update(tensors)
        result["task"] = self.task
        result["precomputed_context"] = self._load(row, "context")["context"]
        for key, shape in {
            "precomputed_input_latents": (1, 16, 21, 60, 104),
            "precomputed_y": (1, 20, 21, 60, 104),
            "precomputed_context": (1, 512, 4096),
        }.items():
            if result[key].shape != shape or result[key].dtype != torch.bfloat16:
                raise ValueError(f"Invalid cached {key}")
        result.update(
            precomputed_height=480,
            precomputed_width=832,
            precomputed_num_frames=81,
            precomputed_source_index=row["source_index"],
        )
        return result


def load_dataset(manifest):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    if data.get("schema") != "uniaction-training-caches":
        raise ValueError("Unknown training cache manifest")
    entries = data["caches"]
    if [e["name"] for e in entries] != list(TASKS):
        raise ValueError("The cache manifest must follow the task stream order")
    parts = []
    for entry in entries:
        root = Path(entry["path"]).expanduser()
        if not root.is_absolute():
            root = manifest.parent / root
        ds = CachedExamples(
            root,
            name=entry["name"],
        )
        if len(ds) != COUNTS[entry["name"]]:
            raise ValueError(f"Unexpected sample count for {entry['name']}")
        parts.append(ds)
    empty = Path(data["empty_prompt_context"]).expanduser()
    if not empty.is_absolute():
        empty = manifest.parent / empty
    if not empty.is_file():
        raise FileNotFoundError(empty)
    return ConcatDataset(parts), empty, data


def single_example(batch):
    if len(batch) != 1:
        raise ValueError("Use one sample per GPU and gradient accumulation")
    return batch[0]
