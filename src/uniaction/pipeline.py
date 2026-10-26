from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import torch
from PIL import Image

from .camera import generate_camera_sample, validate_camera_condition
from .config import make_config
from .inference import build_pipeline, generate_sample
from .media import save_video

DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


class UniActionPipeline:
    """Generate videos from an initial image, scene text and action condition."""

    def __init__(self, config, pipe, adapter):
        self.config, self.pipe, self.adapter = config, pipe, adapter

    @classmethod
    def from_pretrained(
        cls,
        base_model,
        checkpoint_dir,
        *,
        device="cuda",
        attention_backend="fa4",
        cache_dir=None,
        trained=False,
    ):
        """Load the local Wan backbone and UniAction checkpoints."""
        config = make_config(base_model, checkpoint_dir, trained=trained)
        if attention_backend not in ("fa4", "sdpa"):
            raise ValueError("attention_backend must be fa4 or sdpa")
        kwargs = {}
        if attention_backend == "sdpa":
            kwargs = dict(
                allow_compatibility_runtime=True,
                attention_backend="legacy",
                attention_fallback=False,
                rope_backend="baseline",
                rmsnorm_backend="baseline",
            )
        pipe, adapter = build_pipeline(config, device=device, cache_dir=cache_dir, **kwargs)
        return cls(config, pipe, adapter)

    @torch.no_grad()
    def generate(
        self,
        image,
        prompt,
        condition,
        *,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=1234,
        steps=20,
        text_guidance=1.0,
        action_guidance=2.0,
        guidance="separate",
        output=None,
        overlay=False,
    ):
        """Return 81 RGB frames, optionally saving raw and condition-overlay videos."""
        overlay_output = None
        if overlay:
            if output is None:
                raise ValueError("Provide output to save a condition overlay")
            from .overlay import save_overlay

            target = Path(output)
            overlay_output = target.with_name(target.stem + "-overlay.mp4")
            if target.exists() or overlay_output.exists():
                raise ValueError("Raw or overlay output already exists; choose a new output path")

        if isinstance(image, (str, Path)):
            with Image.open(image) as handle:
                image = handle.convert("RGB")
        else:
            image = image.convert("RGB")
        if image.size != (832, 480):
            raise ValueError(
                "Resize/crop the input to 832x480 and express action coordinates in that image"
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty scene description")
        if not isinstance(negative_prompt, str):
            raise ValueError("negative_prompt must be a string")
        for value in (text_guidance, action_guidance):
            if not math.isfinite(value) or value < 1:
                raise ValueError("Guidance scales must be finite and at least one")
        if guidance not in ("separate", "joint", "counterfactual"):
            raise ValueError("guidance must be separate, joint or counterfactual")
        if guidance in ("joint", "counterfactual") and not math.isclose(
            text_guidance, action_guidance
        ):
            raise ValueError("Joint guidance requires equal text and action scales")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        config = replace(
            self.config, evaluation=replace(self.config.evaluation, steps=steps, seed=seed)
        )
        config.validate()
        sample = dict(condition)
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            text_cfg_scale=float(text_guidance),
            action_cfg_scale=float(action_guidance),
        )
        if guidance == "counterfactual":
            from .guidance import generate_counterfactual

            frames = generate_counterfactual(
                config, self.pipe, self.adapter, sample, image, **kwargs
            )
        elif sample.get("semantic_task") == "camera":
            validate_camera_condition(sample)
            mode = "camera_joint" if guidance == "joint" else "separate"
            frames = generate_camera_sample(
                config, self.pipe, self.adapter, sample, image, cfg_mode=mode, **kwargs
            )
        else:
            if sample.get("task") not in ("hand", "force", "robot"):
                raise ValueError("Use an action from uniaction.conditions")
            frames = generate_sample(
                config, self.pipe, self.adapter, sample, image, cfg_mode=guidance, **kwargs
            )
        if len(frames) != 81 or any(frame.size != (832, 480) for frame in frames):
            raise RuntimeError("Generated video does not match the model output specification")
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            save_video(frames, output, fps=16, quality=9)
            if overlay_output is not None:
                save_overlay(output, condition, overlay_output)
        return frames
