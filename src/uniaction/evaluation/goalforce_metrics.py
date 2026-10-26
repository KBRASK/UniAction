from __future__ import annotations
import math
from typing import Any
import cv2
import numpy as np

GOALFORCE_STREAMS = ("goalforce_balls", "goalforce_dominos")
RESPONSE_INTERVALS = {"goalforce_balls": 27, "goalforce_dominos": 27}


def response_intervals_for(stream: str, num_frames: int) -> int:
    if num_frames != 81:
        raise ValueError("Force evaluation requires 81 frames")
    try:
        return RESPONSE_INTERVALS[stream]
    except KeyError as error:
        raise ValueError(f"unsupported GoalForce stream {stream!r}") from error


def _dense_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    if hasattr(cv2, "DISOpticalFlow_create"):
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        estimator.setFinestScale(1)
        return estimator.calc(previous_gray, current_gray, None)
    return cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )


def _fit_background_affine(
    flow: np.ndarray, *, excluded_center: np.ndarray, excluded_radius: float, grid_step: int = 16
) -> np.ndarray:
    height, width = flow.shape[:2]
    y, x = np.mgrid[grid_step // 2 : height : grid_step, grid_step // 2 : width : grid_step]
    source = np.stack([x.reshape(-1), y.reshape(-1)], axis=1).astype(np.float32)
    vectors = flow[y, x].reshape(-1, 2)
    valid = np.isfinite(vectors).all(axis=1)
    valid &= np.linalg.norm(source - excluded_center[None], axis=1) >= excluded_radius
    source = source[valid]
    vectors = vectors[valid]
    if len(source) < 12:
        translation = np.median(vectors, axis=0) if len(vectors) else np.zeros(2)
        return np.asarray(
            [[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]], dtype=np.float64
        )
    affine, _ = cv2.estimateAffinePartial2D(
        source,
        source + vectors,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=1000,
        confidence=0.995,
        refineIters=5,
    )
    if affine is None or not np.isfinite(affine).all():
        translation = np.median(vectors, axis=0)
        affine = np.asarray(
            [[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]], dtype=np.float64
        )
    return np.asarray(affine, dtype=np.float64)


def _affine_displacement(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    source = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=np.float64)], axis=-1)
    return source @ affine.T - points


def _local_residual_motion(
    flow: np.ndarray, affine: np.ndarray, center: np.ndarray, *, radius: float
) -> tuple[np.ndarray, float]:
    height, width = flow.shape[:2]
    y0 = max(0, int(math.floor(center[1] - radius)))
    y1 = min(height, int(math.ceil(center[1] + radius + 1)))
    x0 = max(0, int(math.floor(center[0] - radius)))
    x1 = min(width, int(math.ceil(center[0] + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return (np.zeros(2, dtype=np.float64), 0.0)
    y, x = np.mgrid[y0:y1, x0:x1]
    disk = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    points = np.stack([x[disk], y[disk]], axis=1).astype(np.float64)
    vectors = flow[y[disk], x[disk]].astype(np.float64)
    finite = np.isfinite(vectors).all(axis=1)
    points, vectors = (points[finite], vectors[finite])
    if len(vectors) < 16:
        return (np.zeros(2, dtype=np.float64), 0.0)
    residual = vectors - _affine_displacement(affine, points)
    magnitudes = np.linalg.norm(residual, axis=1)
    cutoff = max(0.2, float(np.quantile(magnitudes, 0.6)))
    moving = residual[magnitudes >= cutoff]
    if len(moving) < 8:
        return (np.zeros(2, dtype=np.float64), 0.0)
    median = np.median(moving, axis=0)
    deviations = np.linalg.norm(moving - median, axis=1)
    inliers = moving[deviations <= max(0.5, float(np.quantile(deviations, 0.7)))]
    if len(inliers):
        median = np.median(inliers, axis=0)
    selected = inliers if len(inliers) else moving
    coherence = float(
        np.linalg.norm(np.mean(selected, axis=0))
        / max(np.mean(np.linalg.norm(selected, axis=1)), 1e-06)
    )
    return (median.astype(np.float64), float(np.clip(coherence, 0.0, 1.0)))


def _score_displacements(
    displacements: np.ndarray, expected_direction: np.ndarray, *, motion_threshold_px: float
) -> dict[str, Any]:
    displacement_array = np.asarray(displacements, dtype=np.float64)
    expected = np.asarray(expected_direction, dtype=np.float64)
    expected /= max(float(np.linalg.norm(expected)), 1e-12)
    magnitudes = np.linalg.norm(displacement_array, axis=1)
    peak_index = int(np.argmax(magnitudes))
    peak = displacement_array[peak_index]
    peak_magnitude = float(magnitudes[peak_index])
    coverage = float(peak_magnitude >= motion_threshold_px)
    cosine = (
        float(np.dot(peak, expected) / peak_magnitude)
        if coverage > 0.0 and peak_magnitude > 1e-06
        else None
    )
    return {
        "motion_coverage": coverage,
        "response_motion_px": peak_magnitude,
        "response_peak_frame": peak_index,
        "direction_cosine": cosine,
        "forward_motion_px": float(np.dot(peak, expected)),
        "lateral_motion_px": float(abs(peak[0] * expected[1] - peak[1] * expected[0])),
        "score": float(np.clip(cosine, 0.0, 1.0)) if cosine else 0.0,
    }


def trace_force_response(
    frames: np.ndarray,
    *,
    start_xy: np.ndarray,
    response_intervals: int,
    roi_radius_px: float = 26.0,
) -> dict[str, Any]:
    center = np.asarray(start_xy, dtype=np.float64).copy()
    cumulative = np.zeros(2, dtype=np.float64)
    displacements = [cumulative.copy()]
    centers = [center.copy()]
    coherences: list[float] = []
    valid_intervals = 0
    interval_count = min(len(frames) - 1, response_intervals)
    for frame_index in range(interval_count):
        flow = _dense_flow(frames[frame_index], frames[frame_index + 1])
        affine = _fit_background_affine(
            flow, excluded_center=center, excluded_radius=3.0 * roi_radius_px
        )
        residual, coherence = _local_residual_motion(flow, affine, center, radius=roi_radius_px)
        background = _affine_displacement(affine, center.reshape(1, 2))[0]
        center += background + residual
        cumulative += residual
        displacements.append(cumulative.copy())
        centers.append(center.copy())
        coherences.append(coherence)
        if coherence >= 0.2:
            valid_intervals += 1
        height, width = frames[0].shape[:2]
        if not -roi_radius_px <= center[0] <= width + roi_radius_px:
            break
        if not -roi_radius_px <= center[1] <= height + roi_radius_px:
            break
    return {
        "displacements": np.asarray(displacements),
        "centers": np.asarray(centers),
        "coherences": np.asarray(coherences),
        "requested_intervals": interval_count,
        "observed_intervals": len(coherences),
        "flow_valid_fraction": valid_intervals / float(max(1, interval_count)),
        "flow_coherence_mean": float(np.mean(coherences)) if coherences else 0.0,
    }


def measure_force_response(
    frames: np.ndarray,
    *,
    start_xy: np.ndarray,
    expected_direction: np.ndarray,
    response_intervals: int,
    roi_radius_px: float = 26.0,
    motion_threshold_px: float = 4.0,
) -> dict[str, Any]:
    trace = trace_force_response(
        frames,
        start_xy=start_xy,
        response_intervals=response_intervals,
        roi_radius_px=roi_radius_px,
    )
    result = _score_displacements(
        trace["displacements"], expected_direction, motion_threshold_px=motion_threshold_px
    )
    result.update({name: trace[name] for name in ("flow_valid_fraction", "flow_coherence_mean")})
    return result
