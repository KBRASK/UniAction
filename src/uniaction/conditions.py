from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .camera import held_camera_condition

HAND_POINTS = (
    "left_wrist",
    "left_thumb_tip",
    "left_index_tip",
    "left_middle_tip",
    "left_ring_tip",
    "left_little_tip",
    "right_wrist",
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_little_tip",
)
ROBOT_POINTS = (
    "base",
    "base_to_joint1_midpoint",
    "joint1",
    "joint2_to_joint3_midpoint",
    "joint3",
    "joint4",
    "joint4_to_joint5_midpoint",
    "joint5",
    "joint7",
    "flange",
    "finger_positive_y",
    "finger_negative_y",
)


def _uv(value, shape, name):
    points = torch.as_tensor(value, dtype=torch.float32).clone()
    if tuple(points.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(points.shape)}")
    if not torch.isfinite(points).all() or (points < 0).any() or (points > 1).any():
        raise ValueError(f"{name} must contain finite normalized (x, y) coordinates in [0, 1]")
    return points


def force(start, end):
    """Create a force from normalized contact and endpoint coordinates."""

    uv = _uv([start, end], (2, 2), "force points")
    if torch.equal(uv[0], uv[1]):
        raise ValueError("A force needs distinct contact and endpoint coordinates")
    return dict(
        task="force",
        semantic_task="force",
        action_token_uv=uv,
        action_token_time=torch.zeros(2),
        action_token_id=torch.tensor([0, 3]),
    )


def camera(key: str):
    """Hold one WASD/IJKL camera command for the video."""

    return held_camera_condition(
        {"camera_event_times": np.arange(80, dtype=np.float32) + 0.5}, key.upper()
    )


def hand_pose(uv, visible=None):
    """Create an 81-frame hand condition in HAND_POINTS order."""

    points = torch.as_tensor(uv, dtype=torch.float32).clone()
    if tuple(points.shape) != (81, 12, 2):
        raise ValueError("Hand pose must have shape (81, 12, 2)")
    valid = torch.isfinite(points).all(-1) & (points >= 0).all(-1) & (points <= 1).all(-1)
    if visible is not None:
        mask = torch.as_tensor(visible)
        if mask.dtype != torch.bool or tuple(mask.shape) != (81, 12):
            raise ValueError("Hand visibility must be a boolean (81, 12) array")
        valid &= mask
    if not valid.any():
        raise ValueError("Hand pose contains no visible points")

    action = torch.zeros(81, 16, 2)
    action[:, :12] = torch.where(valid[..., None], points, 0)
    point_mask = torch.zeros(81, 16, dtype=torch.bool)
    point_mask[:, :12] = valid
    dim_mask = torch.zeros(32, dtype=torch.bool)
    dim_mask[:24] = True
    return dict(
        task="hand",
        semantic_task="hand",
        action=action.reshape(81, 32),
        action_time_mask=torch.ones(81, dtype=torch.bool),
        action_dim_mask=dim_mask,
        action_point_time_mask=point_mask,
    )


def robot_pose(uv, visible=None):
    """Create an 81-frame robot condition in ROBOT_POINTS order."""

    points = torch.as_tensor(uv, dtype=torch.float32).clone()
    if tuple(points.shape) != (81, 12, 2):
        raise ValueError("Robot pose must have shape (81, 12, 2)")
    valid = torch.isfinite(points).all(-1) & (points >= 0).all(-1) & (points <= 1).all(-1)
    if visible is not None:
        mask = torch.as_tensor(visible)
        if mask.dtype != torch.bool or tuple(mask.shape) != (81, 12):
            raise ValueError("Robot visibility must be a boolean (81, 12) array")
        valid &= mask
    if not valid.any():
        raise ValueError("Robot pose contains no visible points")
    return dict(
        task="robot",
        semantic_task="robot",
        action_token_uv=points[valid],
        action_token_time=torch.arange(81, dtype=torch.float32)[:, None].expand(81, 12)[valid],
        action_token_id=torch.arange(12)[None, :].expand(81, 12)[valid],
    )


def load_condition(path: str | Path):
    """Load a force, camera, hand or robot condition from an NPZ file."""

    with np.load(path, allow_pickle=False) as data:
        task = str(data["task"].item())
        if task == "force":
            return force(data["start"], data["end"])
        if task == "camera":
            return camera(str(data["key"].item()))
        if task == "hand":
            return hand_pose(data["uv"], data["visible"] if "visible" in data else None)
        if task == "robot":
            return robot_pose(data["uv"], data["visible"] if "visible" in data else None)
        raise ValueError(f"Unknown action task {task!r}; use force, camera, hand or robot")
