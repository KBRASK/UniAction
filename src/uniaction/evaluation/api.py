from pathlib import Path

import numpy as np
import torch

from .camera_direction import score_direction
from .camera_metrics import summarize_pose
from .goalforce_metrics import measure_force_response, response_intervals_for
from .goalforce_response_profiles import ResponseProfile, measure_profiled_force_response
from .noise_tolerance import score as score_trajectory
from .similarity_metrics import read_rgb_video, _lpips_metric


# Fitted on training trajectories.
TRAJECTORY_TOLERANCE_PX = {"hand": 59.3427821758677, "robot": 42.666721291364986}


def frames(value):
    result = read_rgb_video(value) if isinstance(value, (str, Path)) else np.asarray(value)
    if result.shape != (81, 480, 832, 3) or result.dtype != np.uint8:
        raise ValueError("Expected 81 RGB uint8 frames at 832 x 480")
    return result


class ReconstructionEvaluator:
    """Measure LPIPS over the 80 generated frames."""

    def __init__(self, device="cuda"):
        import lpips

        self.device = torch.device(device)
        self.model = lpips.LPIPS(net="alex", version="0.1").to(self.device).eval()

    def __call__(self, generated, reference):
        actual, expected = frames(generated)[1:], frames(reference)[1:]
        with (
            torch.backends.cudnn.flags(enabled=True, benchmark=False, allow_tf32=False),
            torch.autocast(device_type=self.device.type, enabled=False),
        ):
            distance = _lpips_metric(self.model, expected, actual, device=self.device, batch_size=4)
        return dict(
            lpips=distance,
            first_frame=1,
            last_frame=80,
            frames=80,
            lpips_network="alex",
            lpips_version="0.1",
            lpips_precision="float32",
            lpips_tf32=False,
        )


class ForceEvaluator:
    """Measure force direction following with a score in [0, 1]."""

    def __init__(self, flower_calibration=None):
        import cv2

        cv2.setNumThreads(1)
        self.flower = None
        if flower_calibration is not None:
            from .flower_detector import FlowerDetector

            self.flower = FlowerDetector(flower_calibration)

    def __call__(self, generated, *, start, end, kind, reference_image=None):
        video = frames(generated)
        uv = np.asarray([start, end], dtype=float)
        if uv.shape != (2, 2) or not np.isfinite(uv).all() or (uv < 0).any() or (uv > 1).any():
            raise ValueError("Force endpoints must be normalized coordinates in [0, 1]")
        xy = uv * [832.0, 480.0]
        direction = xy[1] - xy[0]
        if not np.linalg.norm(direction):
            raise ValueError("Force direction must be nonzero")
        if kind == "ball":
            # A local disk measures object motion; the displacement gate rejects flow noise.
            return measure_force_response(
                video,
                start_xy=xy[0],
                expected_direction=direction,
                response_intervals=response_intervals_for("goalforce_balls", 81),
                roi_radius_px=26.0,
                motion_threshold_px=4.0,
            )
        if kind == "domino":
            # Stop at the initial local response before chain-reaction motion dominates.
            profile = ResponseProfile("initial-response", "goalforce_dominos", 40.0, 8.0, 4.0)
            result = measure_profiled_force_response(
                video, start_xy=xy[0], expected_direction=direction, profile=profile
            )
            return result["response"]
        if kind == "flower":
            if reference_image is None:
                raise ValueError("Flower evaluation requires its original input image")
            if self.flower is None:
                from .flower_detector import FlowerDetector

                self.flower = FlowerDetector(Path(__file__).parents[1] / "assets/flower/model.json")
            if np.asarray(reference_image).shape != (480, 832, 3):
                raise ValueError("Flower reference image must be 832 x 480 RGB")
            return self.flower.score_frames(
                video,
                reference_rgb=np.asarray(reference_image),
                start_xy=xy[0],
                requested_direction=direction,
            )["response"]
        raise ValueError("Force kind must be ball, domino or flower")


def trajectory(measured_px, measured_valid, command_uv, command_valid, *, task, static=False):
    """Measure pose following, assigning zero to missing tracks."""
    if task not in TRAJECTORY_TOLERANCE_PX:
        raise ValueError("Trajectory task must be hand or robot")
    command = np.asarray(command_uv, dtype=float)
    mask = np.asarray(command_valid, dtype=bool)
    if command.shape != (81, 12, 2) or mask.shape != (81, 12):
        raise ValueError("Pose conditions require (81, 12, 2) coordinates and (81, 12) visibility")
    mask = mask & np.isfinite(command).all(-1) & (command >= 0).all(-1) & (command <= 1).all(-1)
    result = score_trajectory(
        measured_px,
        measured_valid,
        command * [832.0, 480.0],
        mask,
        tolerance_px=TRAJECTORY_TOLERANCE_PX[task],
        cap_fraction=0.25,
        command_min=4.0,
        measured_min=2.0,
        static_command=static,
    )
    return dict(score=result["score_lower"], **result)


def camera(pose, null_pose, key):
    held = summarize_pose(pose, expected_frames=81)
    reference = summarize_pose(null_pose, expected_frames=81)
    return score_direction(held, reference, key)
