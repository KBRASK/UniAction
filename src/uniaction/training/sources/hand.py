from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from .common import load_video, read_records, source_path, strict_tail_resampled_indices


_ACTION_JOINTS = (
    "leftHand",
    "leftThumbTip",
    "leftIndexFingerTip",
    "leftMiddleFingerTip",
    "leftRingFingerTip",
    "leftLittleFingerTip",
    "rightHand",
    "rightThumbTip",
    "rightIndexFingerTip",
    "rightMiddleFingerTip",
    "rightRingFingerTip",
    "rightLittleFingerTip",
)


def _project_to_uv(positions_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    projected = (np.asarray(intrinsics, dtype=np.float64) @ positions_camera.T).T
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)


class HandDataset(Dataset):
    def __init__(self, root, metadata):
        self.base_path = Path(root)
        self.records = read_records(metadata)
        self.num_frames, self.height, self.width, self.target_fps = 81, 480, 832, 16
        self.to_tensor, self.to_pil = transforms.ToTensor(), transforms.ToPILImage()

    def __len__(self):
        return len(self.records)

    def _action_uv(self, hdf5_path: Path) -> tuple[np.ndarray, np.ndarray]:
        projected: list[np.ndarray] = []
        positive_depth: list[np.ndarray] = []
        with h5py.File(hdf5_path, "r") as handle:
            camera = np.asarray(handle["/transforms/camera"])
            intrinsics = np.asarray(handle["/camera/intrinsic"])
            if intrinsics.ndim == 3:
                intrinsics = intrinsics[0]
            inverse_camera = np.linalg.inv(camera)
            for joint in _ACTION_JOINTS:
                transform = np.asarray(handle[f"/transforms/{joint}"])
                camera_transform = np.matmul(inverse_camera, transform)
                positions_camera = camera_transform[:, :3, 3]
                projected.append(_project_to_uv(positions_camera, intrinsics))
                positive_depth.append(
                    np.isfinite(positions_camera).all(axis=1) & (positions_camera[:, 2] > 1e-8)
                )
        return np.concatenate(projected, axis=1), np.stack(positive_depth, axis=1)

    def _spatial_transform(
        self,
        frames,
        action: np.ndarray,
        positive_depth: np.ndarray | None = None,
    ):
        original_width, original_height = frames[0].size
        scale = max(self.width / original_width, self.height / original_height)
        scaled_height = round(original_height * scale)
        scaled_width = round(original_width * scale)
        crop_top = (scaled_height - self.height) / 2
        crop_left = (scaled_width - self.width) / 2

        frames_out = []
        for offset in range(0, len(frames), 4):
            pixels = torch.stack([self.to_tensor(frame) for frame in frames[offset : offset + 4]])
            pixels = 2 * pixels - 1
            pixels = torchvision.transforms.functional.resize(
                pixels,
                (scaled_height, scaled_width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            pixels = torchvision.transforms.functional.center_crop(
                pixels, (self.height, self.width)
            )
            frames_out.extend(self.to_pil(frame) for frame in (pixels + 1) / 2)
            del pixels

        uv = torch.as_tensor(action).float().reshape(self.num_frames, -1, 2)
        uv[..., 0] = (uv[..., 0] * scale - crop_left) / self.width
        uv[..., 1] = (uv[..., 1] * scale - crop_top) / self.height
        if positive_depth is None:
            raise ValueError("Hand visibility-aware mode requires depth validity")
        depth = torch.as_tensor(positive_depth, dtype=torch.bool)
        expected = (self.num_frames, uv.shape[1])
        if tuple(depth.shape) != expected:
            raise ValueError(f"Hand depth-valid mask must be {expected}, got {tuple(depth.shape)}")
        point_time_mask = (
            depth
            & torch.isfinite(uv).all(dim=-1)
            & (uv >= 0.0).all(dim=-1)
            & (uv <= 1.0).all(dim=-1)
        )
        if not bool(point_time_mask.any()):
            raise ValueError("Hand sample has no visible action points")
        uv = torch.clamp(uv, 0.0, 1.0)

        action = uv.reshape(self.num_frames, -1).to(torch.bfloat16).to(torch.float32)
        return frames_out, action, point_time_mask

    def __getitem__(self, index):
        row = self.records[index]
        frames, fps = load_video(
            source_path(self.base_path, row["video"]),
            retain_indices=lambda count, rate: strict_tail_resampled_indices(
                count, rate, 16, 81, dataset_name="Hand"
            ),
        )
        action, depth = self._action_uv(source_path(self.base_path, row["annotation"]))
        if len(frames) != len(action):
            raise ValueError("Hand video and annotations must have matching frame counts")
        indices = strict_tail_resampled_indices(len(frames), fps, 16, 81, dataset_name="Hand")
        frames, action, visible = self._spatial_transform(
            [frames[int(i)] for i in indices], action[indices], depth[indices]
        )
        return dict(
            video=frames,
            prompt=row["prompt"],
            semantic_task="hand",
            action=action,
            action_time_mask=torch.ones(81, dtype=torch.bool),
            action_dim_mask=torch.ones(action.shape[1], dtype=torch.bool),
            action_point_time_mask=visible,
            action_normalization="none",
        )
