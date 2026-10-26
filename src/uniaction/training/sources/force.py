from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .common import load_video, read_records, source_path, uniformly_sample_indices


def endpoint_geometry(
    *,
    x_start: float,
    y_start: float,
    angle_degrees: float,
    magnitude_normalized: float,
    width: int,
    height: int,
    min_length_fraction: float = 0.0,
    max_length_fraction: float = 0.25,
) -> tuple[float, float, float, float, float]:

    angle = math.radians(float(angle_degrees))
    dx = math.cos(angle)
    dy = -math.sin(angle)
    candidates: list[float] = []
    epsilon = 1e-6
    if dx > epsilon:
        candidates.append((float(width) - 1.0 - float(x_start)) / dx)
    elif dx < -epsilon:
        candidates.append((0.0 - float(x_start)) / dx)
    if dy > epsilon:
        candidates.append((float(height) - 1.0 - float(y_start)) / dy)
    elif dy < -epsilon:
        candidates.append((0.0 - float(y_start)) / dy)
    positive = [value for value in candidates if value > 0.0 and math.isfinite(value)]
    ray_limit = max(0.0, min(positive)) if positive else 0.0
    magnitude = float(np.clip(magnitude_normalized, 0.0, 1.0))
    minimum = min_length_fraction * float(width)
    maximum = max_length_fraction * float(width)
    desired = minimum + (maximum - minimum) * magnitude
    displacement = min(max(0.0, desired), 0.9 * ray_limit)
    return (
        float(x_start) + displacement * dx,
        float(y_start) + displacement * dy,
        float(displacement),
        float(desired),
        float(ray_limit),
    )


def force_endpoints(start, angle, magnitude):
    width, height = 832, 480
    start = np.asarray(start, dtype=np.float64)
    x, y, _, _, _ = endpoint_geometry(
        x_start=float(start[0] * width),
        y_start=float(start[1] * height),
        angle_degrees=angle,
        magnitude_normalized=magnitude,
        width=width,
        height=height,
    )
    uv = np.clip(np.asarray([start, [x / width, y / height]], dtype=np.float32), 0.0, 1.0)
    anchor = uv[0].astype(np.float64) * [width, height]
    bounds = np.array([width - 1.0, height - 1.0])
    if (anchor < 0).any() or (anchor > bounds).any():
        raise ValueError("Force anchor is outside pixel bounds")
    length = float(np.linalg.norm((uv[1].astype(np.float64) - uv[0]) * [width, height]))
    cells = np.minimum(
        np.floor(uv.astype(np.float64) * [width, height] / 16).astype(np.int64),
        np.asarray([width, height]) // 16 - 1,
    )
    if length < 16.0 or np.array_equal(cells[0], cells[1]):
        direction = np.array([math.cos(math.radians(angle)), -math.sin(math.radians(angle))])
        ray_limits, exits = [], []
        for axis, component in enumerate(direction):
            if abs(component) < 1e-12:
                continue
            positive = component > 0
            border = bounds[axis] if positive else 0.0
            ray_limits.append((border - anchor[axis]) / component)
            edge = (math.floor(anchor[axis] / 16) + int(positive)) * 16
            exits.append(max(0.0, (edge - anchor[axis]) / component))
        required = max(16.0, min(exits)) + 0.001
        if required > 0.9 * min(ray_limits):
            raise ValueError("Force endpoint cannot fit inside the frame")
        uv[1] = (anchor + max(length, required) * direction) / [width, height]
    return torch.from_numpy(uv)


class ForceDataset(Dataset):
    def __init__(self, root, metadata):
        self.root = Path(root)
        self.records = read_records(metadata)
        values = np.asarray([float(row["force"]) for row in self.records], dtype=np.float64)
        self.minimum, self.maximum = float(values.min()), float(values.max())
        self.to_tensor, self.to_pil = transforms.ToTensor(), transforms.ToPILImage()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        frames, _ = load_video(
            source_path(self.root, row["video"]),
            retain_indices=lambda count, fps: uniformly_sample_indices(count, 81),
        )
        frames = [frames[int(i)] for i in uniformly_sample_indices(len(frames), 81)]
        x, y = float(row["x"]) / 832, float(row["y"]) / 480
        crop = None
        if row.get("crop_width"):
            left, top, width, height = [
                int(row[key]) for key in ("crop_left", "crop_top", "crop_width", "crop_height")
            ]
            crop = left, top, width, height
            pixel_x = float(row["x"])
            pixel_y = frames[0].height - float(row["y"])
            final_x = ((pixel_x - left) / width) * 832
            final_y = ((pixel_y - top) / height) * 480
            x = float(np.clip(final_x / 832, 0.0, 1.0))
            y = float(np.clip(1.0 - final_y / 480, 0.0, 1.0))
        output = []
        for offset in range(0, 81, 4):
            pixels = torch.stack([self.to_tensor(frame) for frame in frames[offset : offset + 4]])
            pixels = 2 * pixels - 1
            if crop is not None:
                left, top, width, height = crop
                pixels = pixels[:, :, top : top + height, left : left + width]
                pixels = transforms.functional.resize(pixels, [480, 832], antialias=True)
            if tuple(pixels.shape[-2:]) != (480, 832):
                raise ValueError("Force frames must be 832 x 480 after cropping")
            output.extend(self.to_pil(frame) for frame in (pixels + 1) / 2)
        magnitude = float(
            np.clip(
                (float(row["force"]) - self.minimum) / max(1e-6, self.maximum - self.minimum),
                0.0,
                1.0,
            )
        )
        uv = force_endpoints(
            np.asarray([x, 1.0 - y], dtype=np.float32), float(row["angle"]), magnitude
        )
        return dict(
            video=output,
            prompt=row["prompt"],
            semantic_task="force",
            action_token_uv=uv,
            action_token_time=torch.zeros(2, dtype=torch.float32),
            action_token_id=torch.tensor([0, 3], dtype=torch.int64),
            action_normalization="token_list",
        )
