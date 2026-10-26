from __future__ import annotations
import math
import numpy as np

AXES = ("tx", "ty", "tz", "pitch", "yaw", "roll")
KEY_AXIS_SIGN = {
    "W": ("tz", 1),
    "A": ("tx", -1),
    "S": ("tz", -1),
    "D": ("tx", 1),
    "I": ("pitch", 1),
    "K": ("pitch", -1),
    "J": ("yaw", -1),
    "L": ("yaw", 1),
}


def rotation_vector(matrix):
    angle = math.acos(float(np.clip((np.trace(matrix) - 1) * 0.5, -1, 1)))
    skew = np.asarray(
        (matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]),
        dtype=np.float64,
    )
    if angle < 1e-08:
        return skew * 0.5
    if math.pi - angle < 1e-05:
        values, vectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = vectors[:, int(np.argmax(values))]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return axis / np.linalg.norm(axis) * angle
    return skew / (2 * math.sin(angle)) * angle


def summarize_pose(path, *, expected_frames):
    if expected_frames != 81:
        raise ValueError("Camera evaluation requires 81 frames")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"data", "inds"}:
            raise ValueError("unexpected ViPE pose tensor inventory")
        poses, inds = (np.asarray(archive["data"], dtype=np.float64), archive["inds"].copy())
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or inds.shape != (len(poses),)
        or (not np.issubdtype(inds.dtype, np.integer))
        or (not len(poses))
        or (not np.isfinite(poses).all())
        or (inds[0] != 0)
        or (inds[-1] >= expected_frames)
        or bool((np.diff(inds) <= 0).any())
    ):
        raise ValueError("invalid ViPE pose shape, values or native frame indices")
    rotations = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-05, rtol=0)
        or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=0.002, rtol=0)
        or (not np.allclose(np.linalg.det(rotations), 1, atol=0.002, rtol=0))
    ):
        raise ValueError("ViPE poses are not rigid camera-to-world transforms")
    anchored = np.linalg.inv(poses[0]) @ poses
    values = np.column_stack(
        [anchored[:, :3, 3], np.stack([rotation_vector(r[:3, :3]) for r in anchored])]
    )
    late = inds >= math.ceil((expected_frames - 1) * 0.75)
    reliable = (
        len(poses) >= math.ceil(expected_frames * 0.8)
        and inds[-1] >= math.floor((expected_frames - 1) * 0.8)
        and (late.sum() >= 4)
    )
    return {
        "reliable": bool(reliable),
        "pose_count": len(poses),
        "frame_indices": inds.tolist(),
        "frame_coverage": len(poses) / expected_frames,
        "late_pose_count": int(late.sum()),
        "late_axis_values": dict(zip(AXES, np.median(values[late], axis=0).tolist()))
        if late.any()
        else None,
    }
