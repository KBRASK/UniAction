from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from uniaction.evaluation import goalforce_metrics as gm


@dataclass(frozen=True)
class ResponseProfile:
    name: str
    stream: str
    roi_radius_px: float
    response_displacement_px: float
    motion_threshold_px: float = 4.0

    def __post_init__(self) -> None:
        if self.stream not in ("goalforce_dominos",):
            raise ValueError(f"unsupported initial-response stream: {self.stream}")
        if not self.name:
            raise ValueError("a response profile must have a nonempty name")
        for field in ("roi_radius_px", "response_displacement_px", "motion_threshold_px"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if self.response_displacement_px < self.motion_threshold_px:
            raise ValueError("response displacement must reach the motion threshold")


def first_response_window(
    displacements: np.ndarray, *, displacement_px: float
) -> tuple[np.ndarray, str]:
    values = np.asarray(displacements, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise ValueError("displacements must be a nonempty N x 2 array")
    if not np.isfinite(values).all() or not np.allclose(values[0], 0.0, atol=1e-10):
        raise ValueError("displacements must be finite and start at zero")
    if not math.isfinite(displacement_px) or displacement_px <= 0:
        raise ValueError("displacement threshold must be finite and positive")
    crossings = np.flatnonzero(np.linalg.norm(values, axis=1) >= displacement_px)
    if len(crossings):
        return (values[: int(crossings[0]) + 1], "first_displacement")
    return (values, "observed_window_end")


def score_response_trace(
    trace: dict[str, Any], *, expected_direction: np.ndarray, profile: ResponseProfile
) -> dict[str, Any]:
    direction = np.asarray(expected_direction, dtype=np.float64).copy()
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("expected direction must be a finite 2-vector")
    if float(np.linalg.norm(direction)) <= 1e-12:
        raise ValueError("expected direction must be nonzero")
    window, stop_reason = first_response_window(
        trace["displacements"], displacement_px=profile.response_displacement_px
    )
    metrics = gm._score_displacements(
        window, direction, motion_threshold_px=profile.motion_threshold_px
    )
    return {
        **metrics,
        "window_start_frame": 0,
        "window_end_frame": len(window) - 1,
        "window_stop_reason": stop_reason,
        "search_requested_intervals": int(trace["requested_intervals"]),
        "search_observed_intervals": int(trace["observed_intervals"]),
        "search_flow_valid_fraction": float(trace["flow_valid_fraction"]),
        "search_flow_coherence_mean": float(trace["flow_coherence_mean"]),
    }


def measure_profiled_force_response(
    frames: np.ndarray,
    *,
    start_xy: np.ndarray,
    expected_direction: np.ndarray,
    profile: ResponseProfile,
) -> dict[str, Any]:
    if frames.ndim != 4 or frames.shape[1:] != (480, 832, 3) or len(frames) < 2:
        raise ValueError("profiles require at least two RGB frames at 832 x 480")
    start = np.asarray(start_xy, dtype=np.float64)
    if start.shape != (2,) or not np.isfinite(start).all():
        raise ValueError("force start must be a finite 2-vector")
    if not (0 <= start[0] <= 832 and 0 <= start[1] <= 480):
        raise ValueError("force start must lie on the image canvas")
    intervals = gm.response_intervals_for(profile.stream, len(frames))
    trace = gm.trace_force_response(
        frames, start_xy=start, response_intervals=intervals, roi_radius_px=profile.roi_radius_px
    )
    candidate = score_response_trace(trace, expected_direction=expected_direction, profile=profile)
    return {
        "schema": "uniaction-goalforce-initial-response-v1",
        "profile": asdict(profile),
        "response": candidate,
    }
