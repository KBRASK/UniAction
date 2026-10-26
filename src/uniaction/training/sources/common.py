from pathlib import Path
from typing import Sequence
import csv
import shutil
import tempfile

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms


def source_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    path.relative_to(root)
    return path


def read_records(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_video(
    path: str | Path,
    *,
    retries: int = 1,
    retain_indices=None,
) -> tuple[list[Image.Image], float | None]:

    last_error = None
    for attempt in range(max(3, int(retries))):
        with tempfile.TemporaryDirectory(prefix="uniaction-video-") as temp:
            local = Path(temp) / "video.mp4"
            shutil.copyfile(path, local)
            capture = cv2.VideoCapture(str(local), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            if not capture.isOpened():
                capture.release()
                last_error = OSError(f"Could not open source video: {path}")
                continue
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_value = fps if np.isfinite(fps) and fps > 0 else None
            keep = None
            if retain_indices is not None and count > 0:
                keep = {int(index) for index in retain_indices(count, fps_value)}
                if not keep or min(keep) < 0 or max(keep) >= count:
                    capture.release()
                    raise ValueError("Retained frame indices are outside the source video")
                keep.add(0)
            frames = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if keep is None or len(frames) in keep:
                        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    else:
                        frames.append(None)
            finally:
                capture.release()
            if frames and (count <= 0 or len(frames) == count):
                return frames, fps_value
            last_error = OSError(f"Incomplete decode for {path}: {len(frames)}/{count}")
            del frames
    raise last_error


def strict_tail_resampled_indices(
    num_source: int,
    source_fps: float | None,
    target_fps: float | None,
    num_frames: int,
    *,
    dataset_name: str,
) -> np.ndarray:

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_source <= 0:
        raise ValueError(
            f"{dataset_name} has no source frames; {num_frames} distinct frames are required"
        )
    if (
        target_fps is None
        or target_fps <= 0
        or source_fps is None
        or not np.isfinite(source_fps)
        or source_fps <= 0
    ):
        if num_source < num_frames:
            raise ValueError(
                f"{dataset_name} has only {num_source} source frames; {num_frames} "
                "distinct frames are required and temporal padding is forbidden"
            )
        indices = np.arange(num_source - num_frames, num_source, dtype=np.int64)
    else:
        source_duration = (num_source - 1) / float(source_fps)
        required_duration = (num_frames - 1) / float(target_fps)
        if source_duration + 1e-9 < required_duration:
            available = int(np.floor(source_duration * float(target_fps) + 1e-9)) + 1
            raise ValueError(
                f"{dataset_name} has only {available} endpoint-aligned frames after "
                f"temporal resampling; {num_frames} distinct frames are required and "
                "temporal padding is forbidden"
            )

        offsets = np.arange(num_frames - 1, -1, -1, dtype=np.float64)
        times = source_duration - offsets / float(target_fps)
        indices = np.rint(times * float(source_fps)).astype(np.int64)
        indices = np.clip(indices, 0, num_source - 1)
        indices[-1] = num_source - 1
    if len(indices) < num_frames:
        raise ValueError(
            f"{dataset_name} has only {len(indices)} frames after temporal "
            f"resampling; {num_frames} distinct frames are required and temporal "
            "padding is forbidden"
        )
    indices = indices[-num_frames:]
    distinct = int(np.unique(indices).size)
    if distinct != num_frames:
        raise ValueError(
            f"{dataset_name} tail window maps to only {distinct} distinct source "
            f"frames; {num_frames} are required and repeated temporal indices are "
            "forbidden"
        )
    return indices


def uniformly_sample_indices(num_source: int, num_frames: int) -> np.ndarray:

    if num_source <= 0:
        raise ValueError("cannot uniformly sample an empty video")
    if num_source < num_frames:
        raise ValueError(
            f"cannot select {num_frames} distinct frames from a {num_source}-frame "
            "video; temporal padding or repeated indices are forbidden"
        )
    if num_source == num_frames:
        return np.arange(num_source, dtype=np.int64)
    indices = np.linspace(0, num_source - 1, num=num_frames, dtype=np.float64)
    indices = np.clip(np.round(indices).astype(np.int64), 0, num_source - 1)
    if int(np.unique(indices).size) != num_frames:
        raise ValueError(
            f"uniform sampling produced repeated temporal indices for "
            f"{num_source} -> {num_frames} frames"
        )
    return indices


def resize_center_crop_frames(
    frames: Sequence[Image.Image],
    *,
    width: int,
    height: int,
    interpolation: torchvision.transforms.InterpolationMode = (
        torchvision.transforms.InterpolationMode.BILINEAR
    ),
) -> list[Image.Image]:
    if not frames:
        raise ValueError("cannot resize an empty frame sequence")
    orig_width, orig_height = frames[0].size
    scale = max(width / orig_width, height / orig_height)
    scaled_width = round(orig_width * scale)
    scaled_height = round(orig_height * scale)
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    result = []
    for offset in range(0, len(frames), 4):
        tensor = torch.stack([to_tensor(frame) for frame in frames[offset : offset + 4]])
        tensor = torchvision.transforms.functional.resize(
            tensor,
            (scaled_height, scaled_width),
            interpolation=interpolation,
        )
        tensor = torchvision.transforms.functional.center_crop(tensor, (height, width))
        result.extend(to_pil(frame) for frame in tensor)
        del tensor
    return result
