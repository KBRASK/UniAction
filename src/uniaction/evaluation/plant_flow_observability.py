from dataclasses import asdict, dataclass
import cv2
import numpy as np
from uniaction.evaluation import goalforce_metrics as gm


@dataclass(frozen=True)
class PlantObservabilityConfig:
    frame_offsets: tuple[int, ...] = (7, 14)
    min_textured_points: int = 24
    min_local_points: int = 8
    residual_threshold_px: float = 1.0
    residual_quantile: float = 0.9


def check_plant_observability(frames, *, anchors_xy, config=PlantObservabilityConfig()):
    frames = np.asarray(frames)
    anchors = np.asarray(anchors_xy, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[1:] != (480, 832, 3) or frames.dtype != np.uint8:
        raise ValueError("expected RGB uint8 frames at 832 x 480")
    if len(frames) <= max(config.frame_offsets) or anchors.shape != (5, 2):
        raise ValueError("missing response frames or five plant anchors")
    gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    mask = np.zeros_like(gray)
    mask[16:-16, 16:-16] = 255
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=1500, qualityLevel=0.01, minDistance=4, mask=mask, blockSize=5
    )
    base = {
        "schema": "uniaction-plant-flow-observability-v1",
        "config": asdict(config),
    }
    if corners is None or len(corners) < config.min_textured_points:
        return {
            **base,
            "observable": False,
            "reason": "insufficient_textured_points",
            "signal_px": 0.0,
            "traces": [],
        }
    source = corners[:, 0].astype(np.float32)
    center = anchors.mean(axis=0)
    radius = float(np.max(np.linalg.norm(anchors - center, axis=1))) + 50
    local = np.linalg.norm(source - center, axis=1) <= radius
    traces = []
    for offset in config.frame_offsets:
        flow = gm._dense_flow(frames[0], frames[offset])
        vectors = cv2.remap(
            flow, source[:, 0].reshape(-1, 1), source[:, 1].reshape(-1, 1), cv2.INTER_LINEAR
        )[:, 0]
        finite = np.isfinite(vectors).all(axis=1)
        if finite.sum() < config.min_textured_points:
            continue
        affine, inliers = cv2.estimateAffinePartial2D(
            source[finite],
            source[finite] + vectors[finite],
            method=cv2.RANSAC,
            ransacReprojThreshold=0.75,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        if affine is None or not np.isfinite(affine).all():
            continue
        residual = vectors - gm._affine_displacement(affine, source)
        selected = finite & local
        if selected.sum() < config.min_local_points:
            continue
        magnitudes = np.linalg.norm(residual[selected], axis=1)
        traces.append(
            {
                "frame_offset": offset,
                "textured_points": int(finite.sum()),
                "local_points": int(selected.sum()),
                "camera_affine": affine.tolist(),
                "rigid_inlier_fraction": float(inliers.mean()),
                "local_residual_px": float(np.quantile(magnitudes, config.residual_quantile)),
            }
        )
    signal = max((r["local_residual_px"] for r in traces), default=0.0)
    available = len(traces) == len(config.frame_offsets)
    observable = available and signal >= config.residual_threshold_px
    return {
        **base,
        "observable": bool(observable),
        "signal_px": signal,
        "traces": traces,
        "reason": "ok"
        if observable
        else "static_or_rigid_camera_motion"
        if available
        else "unobservable_motion",
    }
