from collections.abc import Sequence

import cv2
import numpy as np
import torch
from PIL import Image

INK, WHITE, ACCENT = (12, 20, 28), (255, 255, 255), (255, 222, 64)

_HAND_EDGES = [(0, index) for index in range(1, 6)] + [(6, index) for index in range(7, 12)]


def _numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _hand_overlay(sample: dict, frames: Sequence[Image.Image]) -> list[Image.Image]:
    action = _numpy(sample["action"], np.float32)
    uv = action.reshape(action.shape[0], -1, 2)
    if len(uv) != len(frames):
        raise ValueError(
            f"Hand overlay requires one action row per video frame, got {len(uv)} actions and {len(frames)} frames"
        )
    valid = np.ones(uv.shape[:2], dtype=bool)
    if "action_time_mask" in sample:
        time_mask = _numpy(sample["action_time_mask"], bool).reshape(-1)
        if time_mask.shape != (len(uv),):
            raise ValueError(f"Hand action_time_mask must be {(len(uv),)}, got {time_mask.shape}")
        valid &= time_mask[:, None]
    if "action_dim_mask" in sample:
        dim_mask = _numpy(sample["action_dim_mask"], bool).reshape(-1)
        if dim_mask.shape != (uv.shape[1] * 2,):
            raise ValueError("Hand action_dim_mask must contain one flag per UV dimension")
        valid &= dim_mask.reshape(uv.shape[1], 2).all(axis=-1)[None, :]
    if "action_point_time_mask" in sample:
        point_time_mask = _numpy(sample["action_point_time_mask"], bool)
        if point_time_mask.shape != valid.shape:
            raise ValueError(
                f"Hand action_point_time_mask must be {valid.shape}, got {point_time_mask.shape}"
            )
        valid &= point_time_mask
    output = []
    for frame_index, frame in enumerate(frames):
        image = np.asarray(frame.convert("RGB")).copy()
        height, width = image.shape[:2]
        points = uv[frame_index]
        x = np.clip(np.round(points[:, 0] * (width - 1)).astype(int), 0, width - 1)
        y = np.clip(np.round(points[:, 1] * (height - 1)).astype(int), 0, height - 1)
        for start, end in _HAND_EDGES:
            if not (valid[frame_index, start] and valid[frame_index, end]):
                continue
            cv2.line(
                image,
                (int(x[start]), int(y[start])),
                (int(x[end]), int(y[end])),
                (0, 255, 0),
                2,
                cv2.LINE_8,
            )
        for point_index in range(12):
            if not valid[frame_index, point_index]:
                continue
            cv2.circle(
                image, (int(x[point_index]), int(y[point_index])), 4, (0, 0, 255), -1, cv2.LINE_8
            )
        output.append(Image.fromarray(image))
    return output


def _sparse_arrays(sample: dict):
    return (
        _numpy(sample["action_token_uv"], np.float32).reshape(-1, 2),
        _numpy(sample["action_token_time"], np.float32).reshape(-1),
        _numpy(sample["action_token_id"], np.int64).reshape(-1),
    )


def _robot_overlay(sample: dict, frames: Sequence[Image.Image]) -> list[Image.Image]:
    uv, times, token_ids = _sparse_arrays(sample)
    edges = [(index, index + 1) for index in range(9)] + [(9, 10), (9, 11)]
    colors = {9: (255, 105, 210), 10: (255, 215, 60), 11: (50, 210, 255)}
    default_color = (255, 105, 210)
    rounded_times = np.rint(times).astype(np.int64)
    output = []
    for frame_index, frame in enumerate(frames):
        image = np.asarray(frame.convert("RGB")).copy()
        height, width = image.shape[:2]
        selected = rounded_times == frame_index
        points = {
            int(token_id): (
                int(round(np.clip(point[0] * width, 0, width - 1))),
                int(round(np.clip(point[1] * height, 0, height - 1))),
            )
            for point, token_id in zip(uv[selected], token_ids[selected])
        }
        layer = image.copy()
        for start, end in edges:
            if start in points and end in points:
                cv2.line(
                    layer,
                    points[start],
                    points[end],
                    colors.get(end, default_color),
                    3,
                    cv2.LINE_AA,
                )
        for token_id, point in points.items():
            color = colors.get(token_id, default_color)
            cv2.circle(layer, point, 6, color, -1, cv2.LINE_AA)
            cv2.circle(layer, point, 8, (20, 20, 20), 1, cv2.LINE_AA)
        image = cv2.addWeighted(layer, 0.82, image, 0.18, 0.0)
        output.append(Image.fromarray(image))
    return output


def _force(rgb, geometry):
    out = rgb.copy()
    for first, last in geometry["arrows"]:
        start = np.asarray(geometry["points"][first], dtype=float)
        end = np.asarray(geometry["points"][last], dtype=float)
        distance = np.linalg.norm(end - start)
        a = tuple(np.rint(start).astype(int))
        if distance < 1:
            cv2.circle(out, a, 6, INK, 3, cv2.LINE_AA)
            cv2.circle(out, a, 6, ACCENT, 2, cv2.LINE_AA)
            continue
        unit = (end - start) / distance
        normal = np.array([-unit[1], unit[0]])
        head, width = (min(24.0, distance * 0.4), min(10.0, distance * 0.19))
        shoulder = end - unit * head
        triangle = np.rint([end, shoulder + normal * width, shoulder - normal * width]).astype(
            np.int32
        )
        b = tuple(np.rint(shoulder + unit * 2).astype(int))
        cv2.line(out, a, b, INK, 12, cv2.LINE_AA)
        cv2.polylines(out, [triangle], True, INK, 7, cv2.LINE_AA)
        cv2.line(out, a, b, WHITE, 9, cv2.LINE_AA)
        cv2.polylines(out, [triangle], True, WHITE, 3, cv2.LINE_AA)
        cv2.line(out, a, b, ACCENT, 6, cv2.LINE_AA)
        cv2.fillConvexPoly(out, triangle, ACCENT, cv2.LINE_AA)
        radius = max(2, min(5, int(distance / 5)))
        cv2.circle(out, a, radius + 3, INK, -1, cv2.LINE_AA)
        cv2.circle(out, a, radius + 1, WHITE, -1, cv2.LINE_AA)
        cv2.circle(out, a, radius, ACCENT, -1, cv2.LINE_AA)
    return out
