from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

from uniaction.backbones.wan.loader import load_tokenizer, load_text_encoder, load_vae
from uniaction.backbones.wan.pipeline import WanPipeline
from uniaction.config import ModelConfig
from .data import COUNTS, TASKS, checked_child, source_specs
from .encoding import encode_prompt_context, encode_training_video
from .recovery import atomic_json, atomic_safetensors, inventory_file, validate_inventory


def source_dataset(name, root, metadata):
    from .sources.hand import HandDataset
    from .sources.robot import RobotDataset
    from .sources.force import ForceDataset
    from .sources.camera import CameraDataset

    if name not in TASKS:
        raise ValueError(f"Unknown stream: {name}")
    cls = {
        "hand": HandDataset,
        "robot": RobotDataset,
        "force": ForceDataset,
        "camera": CameraDataset,
    }[TASKS[name]]
    dataset = cls(root, metadata)
    if len(dataset) != COUNTS[name]:
        raise ValueError(f"{name}: incomplete source dataset ({len(dataset)} rows)")
    return dataset


def load_preprocessors(base_model, device):
    base = Path(base_model).resolve()
    model = ModelConfig(base, base, base / "google/umt5-xxl")
    pipe = WanPipeline(torch_dtype=torch.bfloat16, device=device)

    pipe.dit = SimpleNamespace(
        require_vae_embedding=True,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=False,
    )
    pipe.vae = load_vae(model.profile, base / model.vae_file, dtype=torch.bfloat16, device=device)
    pipe.text_encoder = load_text_encoder(base / model.t5_file, dtype=torch.bfloat16, device=device)
    pipe.tokenizer = load_tokenizer(model.tokenizer_root)
    pipe.height_division_factor = pipe.width_division_factor = 16
    pipe.eval().requires_grad_(False)
    return pipe, model


def split_action(sample):

    tensors, fields = {}, {}
    allowed = {
        "prompt",
        "semantic_task",
        "action_normalization",
        "action_token_vocab_size",
        "camera_event_times",
        "camera_layout",
    }
    for name, value in sample.items():
        if name.startswith("action") and isinstance(value, (torch.Tensor, np.ndarray)):
            tensors[name] = torch.as_tensor(value).contiguous().cpu()
        elif name in allowed:
            fields[name] = value.tolist() if isinstance(value, np.ndarray) else value
    return tensors, fields


def encode_stream(args):
    name = args.stream
    metadata = Path(args.metadata).resolve()
    dataset = source_dataset(name, Path(args.source_root), metadata)
    base, root = Path(args.base_model).resolve(), Path(args.output).resolve()
    model = ModelConfig(base, base, base / "google/umt5-xxl")
    config = dict(
        model_family=model.family,
        height=480,
        width=832,
        num_frames=81,
        fps=16,
        stream=name,
        source_metadata=str(metadata),
        seed=1,
        storage="combined-sample-v1",
        encoders=[str(base / model.vae_file), str(base / model.t5_file)],
    )
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists():
        if json.loads((root / "config.json").read_text()) != config:
            raise ValueError("Existing cache has different inputs")
    else:
        atomic_json(root / "config.json", config)

    import fcntl

    with (root / ".encode.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        contexts = {}
        pipe = None
        for index in range(len(dataset)):
            receipt = root / "receipts" / f"{index:05d}.json"
            if receipt.exists():
                row = json.loads(receipt.read_text())
                if row["source_index"] != index:
                    raise ValueError("Cache receipt source index mismatch")
                for key in ("sample", "context"):
                    validate_inventory(root, row[key], dict(size=row[key + "_size"]))
                contexts[row["fields"]["prompt"]] = row["context"]
                records.append(row)
                continue
            if pipe is None:
                pipe, model = load_preprocessors(base, args.device)
            sample = dataset[index]
            prompt = sample["prompt"]
            if prompt in contexts:
                context_name = contexts[prompt]
            else:
                context_name = (
                    f"contexts/{len(contexts):05d}.safetensors"
                    if prompt
                    else "contexts/empty.safetensors"
                )
                atomic_safetensors(
                    checked_child(root, context_name),
                    {"context": encode_prompt_context(pipe, prompt)},
                )
                contexts[prompt] = context_name
            context_path = checked_child(root, context_name)
            context = load_file(str(context_path))["context"]
            encoded = encode_training_video(
                pipe, model, sample["video"], prompt, precomputed_context=context
            )
            encoded.pop("precomputed_context")
            tensors, fields = split_action(sample)
            tensors.update(encoded)
            sample_name = f"samples/{index:05d}.safetensors"
            sample_info = atomic_safetensors(root / sample_name, tensors)
            context_info = inventory_file(context_path)
            row = dict(
                source_index=index,
                fields=fields,
                sample=sample_name,
                sample_size=sample_info["size"],
                context=context_name,
                context_size=context_info["size"],
            )
            atomic_json(receipt, row)
            records.append(row)
            if index % 100 == 0:
                print(f"{name}: {index + 1}/{len(dataset)}", flush=True)
        empty = root / "contexts" / "empty.safetensors"
        if not empty.exists():
            if pipe is None:
                pipe, model = load_preprocessors(base, args.device)
            atomic_safetensors(empty, {"context": encode_prompt_context(pipe, "")})
        atomic_json(
            root / "index.json",
            dict(format="uniaction-training-cache", config=config, records=records),
        )
    print(f"{name}: complete ({len(records)} samples)")


def encode_sources(args):
    prepared = []
    for row in source_specs(args.sources):
        prepared.append(
            SimpleNamespace(
                base_model=args.base_model,
                stream=row["name"],
                source_root=row["source_root"],
                metadata=row["metadata"],
                output=Path(args.output) / row["name"],
                device=args.device,
            )
        )
    for stream in prepared:
        encode_stream(stream)
    print(cache_manifest(args.output, Path(args.output) / "training.json", resume=True))


def cache_manifest(root, output, *, resume=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    entries = []
    import os

    for name in TASKS:
        path = root / name
        index = json.loads((path / "index.json").read_text())
        if len(index["records"]) != COUNTS[name]:
            raise ValueError(f"Incomplete {name} cache")
        entries.append(
            dict(
                name=name,
                path=os.path.relpath(path, output.parent),
            )
        )
    empty = root / "hand/contexts" / "empty.safetensors"
    payload = dict(
        schema="uniaction-training-caches",
        caches=entries,
        empty_prompt_context=os.path.relpath(empty, output.parent),
    )
    if output.exists():
        if resume and json.loads(output.read_text()) == payload:
            return output
        raise FileExistsError(output)
    atomic_json(output, payload)
    return output
