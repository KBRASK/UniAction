from __future__ import annotations
from pathlib import Path
import numpy as np
import torch


def read_rgb_video(path_value: str | Path) -> np.ndarray:
    import cv2

    path = Path(path_value).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video has no decodable frames: {path}")
    return np.stack(frames)


def _lpips_metric(
    model: torch.nn.Module,
    ground_truth: np.ndarray,
    generated: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    values: list[float] = []
    for start in range(0, len(ground_truth), batch_size):
        expected = torch.from_numpy(ground_truth[start : start + batch_size]).to(
            device=device, dtype=torch.float32
        )
        actual = torch.from_numpy(generated[start : start + batch_size]).to(
            device=device, dtype=torch.float32
        )
        expected = expected.permute(0, 3, 1, 2).div_(127.5).sub_(1.0)
        actual = actual.permute(0, 3, 1, 2).div_(127.5).sub_(1.0)
        with torch.inference_mode():
            distance = model(expected, actual, normalize=False)
        values.extend((float(value) for value in distance.flatten().cpu()))
    return float(np.mean(values))
