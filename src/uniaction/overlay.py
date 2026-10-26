from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

try:
    from ._overlay_styles import _robot_overlay, _hand_overlay, _force, _numpy
except ImportError as error:
    raise ImportError("Overlay rendering requires: pip install 'uniaction[overlay]'") from error

from .data.camera_keys import CAMERA_KEYS


@lru_cache(maxsize=9)
def _keyboard(key):
    if key and key not in CAMERA_KEYS:
        raise ValueError("Unknown camera key")
    with Image.open(Path(__file__).parent / "assets" / f"camera-{key or 'none'}.png") as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    return rgba[..., :3], rgba[..., 3:] / 255


def overlay_frames(frames, condition):
    if len(frames) != 81 or any(frame.size != (832, 480) for frame in frames):
        raise ValueError("Overlay expects 81 frames at 832x480")
    task = condition.get("semantic_task")
    if task == "hand":
        marked = _hand_overlay(condition, frames)
        return [
            Image.fromarray(
                np.rint(
                    np.asarray(raw, dtype=np.float64) * 0.30
                    + np.asarray(layer, dtype=np.float64) * 0.70
                ).astype(np.uint8)
            )
            for raw, layer in zip(frames, marked)
        ]
    if task == "robot":
        return _robot_overlay(condition, frames)
    if task == "force":
        uv = _numpy(condition["action_token_uv"], np.float32)
        if uv.shape != (2, 2):
            raise ValueError("Force overlay expects an origin and endpoint")
        points = np.clip(uv * np.array([832, 480]), [0, 0], [831, 479])
        geometry = dict(points={0: points[0], 1: points[1]}, arrows=[(0, 1)])
        return [
            Image.fromarray(_force(np.asarray(frame.convert("RGB")), geometry)) for frame in frames
        ]
    if task == "camera":
        times = _numpy(condition["action_token_time"], np.float32)
        ids = _numpy(condition["action_token_id"], np.int64)
        result = []
        for index, frame in enumerate(frames):
            active = set(ids[(times >= index) & (times < index + 1)].tolist())
            if len(active) > 1:
                raise ValueError("Camera overlay expects a single key per interval")
            key = CAMERA_KEYS[next(iter(active))] if active else ""
            color, alpha = _keyboard(key)
            rgb = np.asarray(frame.convert("RGB"))
            result.append(
                Image.fromarray(np.rint(rgb * (1 - alpha) + color * alpha).astype(np.uint8))
            )
        return result
    raise ValueError(f"Unsupported overlay task: {task!r}")


def save_overlay(video, condition, output):
    video, output = Path(video).resolve(strict=True), Path(output)
    if output.exists() or output.is_symlink() or output.resolve() == video:
        raise ValueError("Overlay output must be new and distinct from the raw video")
    with imageio.get_reader(
        str(video), input_params=["-threads", "2"], output_params=["-threads", "2"]
    ) as reader:
        meta = reader.get_meta_data()
        if tuple(meta["size"]) != (832, 480) or abs(meta["fps"] - 16) > 0.001:
            raise ValueError("Overlay expects an 832x480 video at 16 fps")
        frames = [Image.fromarray(frame) for frame in reader]
    frames = overlay_frames(frames, condition)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb"):
        pass
    try:
        with imageio.get_writer(
            str(output),
            fps=16,
            codec="libx264",
            quality=None,
            macro_block_size=None,
            output_params=[
                "-crf",
                "17",
                "-preset",
                "fast",
                "-threads",
                "2",
                "-movflags",
                "+faststart",
            ],
        ) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output
