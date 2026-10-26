from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    load_video,
    read_records,
    source_path,
    uniformly_sample_indices,
    resize_center_crop_frames,
)


PANDA_MDH = np.asarray(
    [
        [0, 0, 0.333],
        [0, -np.pi / 2, 0],
        [0, np.pi / 2, 0.316],
        [0.0825, np.pi / 2, 0],
        [-0.0825, -np.pi / 2, 0.384],
        [0, np.pi / 2, 0],
        [0.088, np.pi / 2, 0],
    ]
)
FLANGE_D = 0.107
GRIPPER_MAX_WIDTH = 0.08
FINGER_LENGTH = 0.1


def _mdh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    cosine_theta, sine_theta = (np.cos(theta), np.sin(theta))
    cosine_alpha, sine_alpha = (np.cos(alpha), np.sin(alpha))
    return np.asarray(
        [
            [cosine_theta, -sine_theta, 0, a],
            [sine_theta * cosine_alpha, cosine_theta * cosine_alpha, -sine_alpha, -d * sine_alpha],
            [sine_theta * sine_alpha, cosine_theta * sine_alpha, cosine_alpha, d * cosine_alpha],
            [0, 0, 0, 1],
        ]
    )


def panda_fk(joint_angles: np.ndarray, opening: float) -> np.ndarray:
    frames = [np.eye(4)]
    transform = np.eye(4)
    for index in range(7):
        a, alpha, d = PANDA_MDH[index]
        transform = transform @ _mdh_matrix(a, alpha, d, joint_angles[index])
        frames.append(transform.copy())
    flange = transform @ _mdh_matrix(0, 0, FLANGE_D, 0)
    frames.append(flange.copy())
    position = lambda index: frames[index][:3, 3].copy()
    midpoint = lambda left, right: (np.asarray(left) + np.asarray(right)) / 2.0
    points = [
        position(0),
        midpoint(position(0), position(1)),
        position(1),
        midpoint(position(3), position(2)),
        position(3),
        position(4),
        midpoint(position(4), position(5)),
        position(5),
        position(7),
        position(8),
    ]
    half_width = opening * GRIPPER_MAX_WIDTH / 2.0
    for sign in (+1, -1):
        points.append((flange @ np.asarray([0, sign * half_width, FINGER_LENGTH, 1]))[:3])
    return np.asarray(points)


def _intrinsics_matrix(intrinsics: dict) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def _adjust_intrinsics(
    intrinsics: dict, orig_width: int, orig_height: int, width: int, height: int
) -> dict:
    if orig_width == width and orig_height == height:
        return {key: intrinsics[key] for key in ("fx", "fy", "cx", "cy")}
    scale = max(width / orig_width, height / orig_height)
    scaled_width = round(orig_width * scale)
    scaled_height = round(orig_height * scale)
    crop_left = (scaled_width - width) / 2.0
    crop_top = (scaled_height - height) / 2.0
    return {
        "fx": intrinsics["fx"] * scale,
        "fy": intrinsics["fy"] * scale,
        "cx": intrinsics["cx"] * scale - crop_left,
        "cy": intrinsics["cy"] * scale - crop_top,
    }


def _project_visible(
    points: np.ndarray, world_to_camera: np.ndarray, intrinsics: dict, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.hstack([np.asarray(points), np.ones((len(points), 1))])
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    pixels = (_intrinsics_matrix(intrinsics) @ camera.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    uv = np.empty((len(points), 2), dtype=np.float32)
    uv[:, 0] = pixels[:, 0] / float(width)
    uv[:, 1] = pixels[:, 1] / float(height)
    valid = (
        np.isfinite(uv).all(axis=-1)
        & np.isfinite(camera[:, 2])
        & (camera[:, 2] > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= 1.0)
    )
    return (uv, valid)


def build_sparse_tokens(
    action: np.ndarray,
    *,
    dataset_type: str,
    world_to_camera: np.ndarray,
    intrinsics: dict,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    vocab_size = 12
    uv = np.zeros((len(action), vocab_size, 2), dtype=np.float32)
    valid = np.zeros((len(action), vocab_size), dtype=bool)
    openings = 1.0 - np.clip(action[:, 7].astype(np.float32), 0.0, 1.0)
    for frame_index in range(len(action)):
        points = panda_fk(action[frame_index, :7], float(openings[frame_index]))
        uv[frame_index], valid[frame_index] = _project_visible(
            points, world_to_camera, intrinsics, width, height
        )
    ids = np.broadcast_to(np.arange(vocab_size, dtype=np.int64)[None, :], valid.shape)
    times = np.broadcast_to(np.arange(len(action), dtype=np.float32)[:, None], valid.shape)
    if not bool(valid.any()):
        raise ValueError(f"Robot sample has no visible {dataset_type} tokens")
    return (
        torch.from_numpy(uv[valid].astype(np.float32, copy=False)),
        torch.from_numpy(times[valid].astype(np.float32, copy=False)),
        torch.from_numpy(ids[valid].astype(np.int64, copy=False)),
        vocab_size,
    )


class RobotDataset(Dataset):
    def __init__(self, root, metadata):
        self.root = Path(root)
        self.records = read_records(metadata)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        frames, _ = load_video(
            source_path(self.root, row["video"]),
            retain_indices=lambda count, fps: uniformly_sample_indices(count, 81),
        )
        data = json.loads(source_path(self.root, row["annotation"]).read_text())
        action = np.stack(
            [
                np.asarray(
                    list(frame["joint_positions"]) + [frame["gripper_position"]], dtype=np.float32
                )
                for frame in data["frames"]
            ]
        )
        if len(frames) != len(action):
            raise ValueError("Robot video and annotations must have matching frame counts")
        indices = uniformly_sample_indices(len(frames), 81)
        frames = [frames[int(i)] for i in indices]
        width, height = frames[0].size
        frames = resize_center_crop_frames(frames, width=832, height=480)
        intrinsics = _adjust_intrinsics(data["intrinsics"], width, height, 832, 480)
        uv, times, ids, vocabulary = build_sparse_tokens(
            action[indices],
            dataset_type="robot",
            world_to_camera=np.asarray(data["world2cam"], dtype=np.float64),
            intrinsics=intrinsics,
            width=832,
            height=480,
        )
        return dict(
            video=frames,
            prompt=row["prompt"],
            semantic_task="robot",
            action_token_uv=uv,
            action_token_time=times,
            action_token_id=ids,
            action_token_vocab_size=vocabulary,
            action_normalization="token_list",
        )
