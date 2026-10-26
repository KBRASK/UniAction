from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def save_video(
    frames: Iterable[Image.Image | np.ndarray],
    path: str | Path,
    *,
    fps: float,
    quality: int = 9,
    ffmpeg_params: list[str] | None = None,
) -> Path:

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    return output
