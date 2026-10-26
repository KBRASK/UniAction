from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image

from uniaction.config import ModelConfig


_ENCODING_UNITS = {
    "WanVideoUnit_ShapeChecker",
    "WanVideoUnit_NoiseInitializer",
    "WanVideoUnit_PromptEmbedder",
    "WanVideoUnit_InputVideoEmbedder",
    "WanVideoUnit_ImageEmbedderVAE",
    "WanVideoUnit_ImageEmbedderCLIP",
    "WanVideoUnit_ImageEmbedderFused",
}


@torch.no_grad()
def encode_prompt_context(
    pipe,
    prompt: str,
    *,
    output_device: str | torch.device = "cpu",
) -> torch.Tensor:

    matches = [unit for unit in pipe.units if type(unit).__name__ == "WanVideoUnit_PromptEmbedder"]
    if len(matches) != 1:
        raise RuntimeError(
            "DiffSynth pipeline must contain exactly one WanVideoUnit_PromptEmbedder, "
            f"found {len(matches)}"
        )

    shared: dict[str, float | torch.Tensor] = {"cfg_scale": 1.0}
    positive: dict[str, str | torch.Tensor] = {"prompt": str(prompt)}
    negative: dict[str, torch.Tensor] = {}
    _, positive, _ = pipe.unit_runner(matches[0], pipe, shared, positive, negative)
    context = positive.get("context")
    if not isinstance(context, torch.Tensor) or tuple(context.shape) != (1, 512, 4096):
        raise RuntimeError(
            "DiffSynth T5 unit produced invalid context: "
            f"{None if not isinstance(context, torch.Tensor) else tuple(context.shape)}"
        )
    return context.detach().to(device=output_device, dtype=torch.bfloat16).contiguous()


@torch.no_grad()
def encode_training_video(
    pipe,
    model: ModelConfig,
    frames: Sequence[Image.Image],
    prompt: str,
    *,
    output_device: str | torch.device = "cpu",
    precomputed_context: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:

    if len(frames) != model.num_frames:
        raise ValueError(f"expected {model.num_frames} frames, got {len(frames)}")
    bad_sizes = [
        index for index, frame in enumerate(frames) if frame.size != (model.width, model.height)
    ]
    if bad_sizes:
        raise ValueError(
            f"frames must already be {model.width}x{model.height}; bad indices={bad_sizes[:10]}"
        )
    pipe.scheduler.set_timesteps(1000, training=True)
    inputs_shared = {
        "input_video": list(frames),
        "input_image": frames[0],
        "height": model.height,
        "width": model.width,
        "num_frames": model.num_frames,
        "seed": 0,
        "rand_device": pipe.device,
        "cfg_scale": 1.0,
        "tiled": False,
        "tile_size": None,
        "tile_stride": None,
    }
    if precomputed_context is not None:
        if tuple(precomputed_context.shape) != (1, 512, 4096):
            raise ValueError(
                "precomputed_context must have shape (1,512,4096), got "
                f"{tuple(precomputed_context.shape)}"
            )
        inputs_posi = {
            "prompt": prompt,
            "context": precomputed_context.to(
                device=pipe.device,
                dtype=pipe.torch_dtype,
            ),
        }
    else:
        inputs_posi = {"prompt": prompt}
    inputs_nega: dict[str, torch.Tensor] = {}
    seen: set[str] = set()
    for unit in pipe.units:
        name = type(unit).__name__
        if name not in _ENCODING_UNITS:
            continue
        if name == "WanVideoUnit_PromptEmbedder" and precomputed_context is not None:
            continue
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit,
            pipe,
            inputs_shared,
            inputs_posi,
            inputs_nega,
        )
        seen.add(name)
    expected_units = (
        _ENCODING_UNITS - {"WanVideoUnit_PromptEmbedder"}
        if precomputed_context is not None
        else _ENCODING_UNITS
    )
    if seen != expected_units:
        missing_units = expected_units - seen
        raise RuntimeError(f"Wan pipeline is missing cache encoding units: {missing_units}")

    names = ["input_latents", "context", "y"]
    combined = {**inputs_shared, **inputs_posi}
    missing = [name for name in names if not isinstance(combined.get(name), torch.Tensor)]
    if missing:
        raise RuntimeError(f"Wan pipeline did not produce cache tensors: {missing}")
    output = {
        f"precomputed_{name}": (
            combined[name].detach().to(device=output_device, dtype=torch.bfloat16).contiguous()
        )
        for name in names
    }
    return output
