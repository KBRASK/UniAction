from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import cv2
import numpy as np
from uniaction.evaluation import goalforce_metrics as gm


@dataclass(frozen=True)
class PlantFlowConfig:
    response_intervals: int = 14
    radii_px: tuple[float, ...] = (12.0, 26.0, 40.0)
    min_registration_inliers: int = 24
    registration_reprojection_px: float = 2.0
    max_force_point_distance_px: float = 8.0
    coherence_threshold: float = 0.2

    def __post_init__(self):
        object.__setattr__(self, "radii_px", tuple((float(v) for v in self.radii_px)))
        if self.response_intervals < 1 or self.response_intervals > 80:
            raise ValueError("response_intervals must lie in 1..80")
        if not self.radii_px or any((not np.isfinite(v) or v <= 0 for v in self.radii_px)):
            raise ValueError("radii must be finite and positive")
        if self.min_registration_inliers < 3:
            raise ValueError("at least three registration inliers are required")
        for value in (self.registration_reprojection_px, self.max_force_point_distance_px):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("registration distances must be finite and positive")
        if not 0 <= self.coherence_threshold <= 1:
            raise ValueError("coherence threshold must lie in 0..1")


class PlantViewBank:
    def __init__(
        self, reference_rgb: np.ndarray, anchors_xy: np.ndarray, *, config: PlantFlowConfig
    ):
        self.config = config
        self.images = np.asarray(reference_rgb)
        self.anchors = np.asarray(anchors_xy, dtype=np.float64)
        if self.images.shape != (5, 480, 832, 3) or self.images.dtype != np.uint8:
            raise ValueError("view bank must contain five 832 x 480 uint8 RGB images")
        if self.anchors.shape != (5, 5, 2) or not np.isfinite(self.anchors).all():
            raise ValueError("view bank must contain five finite force-point anchors per view")
        self.orb = cv2.ORB_create(nfeatures=3000, edgeThreshold=15, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.descriptors = [
            self.orb.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
            for rgb in self.images
        ]

    def register(self, reference_rgb: np.ndarray, start_xy: np.ndarray) -> dict[str, Any]:
        if reference_rgb.shape != (480, 832, 3) or reference_rgb.dtype != np.uint8:
            raise ValueError("conditioning image must be 832 x 480 uint8 RGB")
        start = np.asarray(start_xy, dtype=np.float64)
        if start.shape != (2,) or not np.isfinite(start).all():
            raise ValueError("force position must be a finite 2-vector")
        query_points, query_descriptors = self.orb.detectAndCompute(
            cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY), None
        )
        if query_descriptors is None:
            return {"valid": False, "reason": "no_conditioning_keypoints"}
        candidates = []
        for view, (points, descriptors) in enumerate(self.descriptors):
            if descriptors is None:
                continue
            pairs = self.matcher.knnMatch(descriptors, query_descriptors, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(good) < self.config.min_registration_inliers:
                continue
            original = np.asarray([points[m.queryIdx].pt for m in good], dtype=np.float64)
            target = np.asarray([query_points[m.trainIdx].pt for m in good], dtype=np.float64)
            cv2.setRNGSeed(0)
            affine, inliers = cv2.estimateAffinePartial2D(
                original,
                target,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.registration_reprojection_px,
                maxIters=2000,
                confidence=0.999,
                refineIters=10,
            )
            if affine is None or not np.isfinite(affine).all():
                continue
            count = int(inliers.sum())
            scale = float(np.sqrt(np.linalg.det(affine[:, :2])))
            if count < self.config.min_registration_inliers or not 0.75 <= scale <= 1.5:
                continue
            residual = original @ affine[:, :2].T + affine[:, 2] - target
            error = float(
                np.sqrt(np.mean(np.sum(residual[inliers[:, 0].astype(bool)] ** 2, axis=1)))
            )
            candidates.append(
                {
                    "view_id": view,
                    "affine": affine.tolist(),
                    "inliers": count,
                    "matches": len(good),
                    "registration_rms_px": error,
                    "scale": scale,
                }
            )
        if not candidates:
            return {"valid": False, "reason": "no_supported_reference_view"}
        best = max(candidates, key=lambda v: (v["inliers"], -v["registration_rms_px"]))
        affine = np.asarray(best["affine"])
        canonical = np.linalg.solve(affine[:, :2], start - affine[:, 2])
        distances = np.linalg.norm(self.anchors[best["view_id"]] - canonical, axis=1)
        point = int(np.argmin(distances))
        best.update(
            {
                "canonical_force_xy": canonical.tolist(),
                "point_id": point,
                "force_point_distance_px": float(distances[point]),
                "group_id": best["view_id"] * 5 + point,
            }
        )
        if distances[point] > self.config.max_force_point_distance_px:
            return {**best, "valid": False, "reason": "force_point_outside_calibrated_anchors"}
        return {**best, "valid": True, "reason": "ok"}


def extract_plant_flow_features(
    frames: np.ndarray, *, reference_rgb: np.ndarray, start_xy: np.ndarray, view_bank: PlantViewBank
) -> dict[str, Any]:
    config = view_bank.config
    if frames.ndim != 4 or frames.shape[1:] != (480, 832, 3) or frames.dtype != np.uint8:
        raise ValueError("frames must be N x 480 x 832 x 3 uint8 RGB")
    if len(frames) <= config.response_intervals:
        raise ValueError("not enough frames for the fixed response window")
    registration = view_bank.register(reference_rgb, start_xy)
    result = {
        "schema": "uniaction-flower-flow-features",
        "config": asdict(config),
        "registration": registration,
    }
    if not registration["valid"]:
        return {**result, "valid": False, "reason": registration["reason"]}
    affine = np.asarray(registration["affine"], dtype=np.float64)
    inverse = np.linalg.inv(affine[:, :2])
    anchors = view_bank.anchors[registration["view_id"]] @ affine[:, :2].T + affine[:, 2]
    count, radius_count = (len(anchors), len(config.radii_px))
    centers = np.repeat(anchors[:, None, :], radius_count, axis=1)
    initial = centers.copy()
    cumulative = np.zeros_like(centers)
    alive = np.ones((count, radius_count), dtype=bool)
    displacement_rows, velocity_rows, mask_rows, coherence_rows, center_rows = (
        [],
        [],
        [],
        [],
        [centers.copy()],
    )
    background_center = np.mean(anchors, axis=0)
    background_radius = (
        float(np.max(np.linalg.norm(anchors - background_center, axis=1)))
        + 40 * registration["scale"]
    )
    for frame in range(config.response_intervals):
        flow = gm._dense_flow(frames[frame], frames[frame + 1])
        camera = gm._fit_background_affine(
            flow, excluded_center=background_center, excluded_radius=background_radius
        )
        velocities = np.zeros_like(centers)
        coherent = np.zeros_like(alive)
        coherences = np.zeros_like(alive, dtype=np.float64)
        for point in range(count):
            for j, radius in enumerate(config.radii_px):
                center = centers[point, j]
                if not alive[point, j] or not (0 <= center[0] < 832 and 0 <= center[1] < 480):
                    alive[point, j] = False
                    continue
                motion, coherence = gm._local_residual_motion(
                    flow, camera, center, radius=radius * registration["scale"]
                )
                background = gm._affine_displacement(camera, center[None])[0]
                centers[point, j] += background + motion
                velocities[point, j] = inverse @ motion
                cumulative[point, j] += velocities[point, j]
                coherences[point, j] = coherence
                coherent[point, j] = coherence >= config.coherence_threshold
        displacement_rows.append(cumulative.copy())
        velocity_rows.append(velocities)
        mask_rows.append(alive & coherent)
        coherence_rows.append(coherences)
        center_rows.append(centers.copy())
        background_center += gm._affine_displacement(camera, background_center[None])[0]
    cumulative_array = np.asarray(displacement_rows)
    valid_array = np.asarray(mask_rows)
    magnitudes = np.linalg.norm(cumulative_array, axis=-1)
    site_peaks = np.max(np.where(valid_array, magnitudes, 0.0), axis=0)
    return {
        **result,
        "valid": True,
        "reason": "ok",
        "initial_centers": initial,
        "cumulative": cumulative_array,
        "velocity": np.asarray(velocity_rows),
        "valid_mask": valid_array,
        "coherence": np.asarray(coherence_rows),
        "centers": np.asarray(center_rows),
        "motion_signal_px": float(np.quantile(site_peaks, 0.8)),
    }


def feature_vector(
    record: dict[str, Any], *, intervals: int = 14, representation: str = "cumulative"
) -> tuple[np.ndarray, np.ndarray]:
    if representation not in ("cumulative", "velocity"):
        raise ValueError("unsupported flow representation")
    values = np.asarray(record[representation], dtype=np.float64)[:intervals]
    valid = np.asarray(record["valid_mask"], dtype=bool)[:intervals]
    if len(values) != intervals or values.shape != valid.shape + (2,):
        raise ValueError("feature/mask shape mismatch")
    if not np.isfinite(values).all():
        raise ValueError("non-finite flow features")
    return (values.reshape(-1), np.repeat(valid[..., None], 2, axis=-1).reshape(-1))


def score_inferred_force(
    prediction: dict[str, Any],
    *,
    requested_direction: np.ndarray,
    canonical_to_image: np.ndarray,
    motion_signal_px: float,
    motion_threshold_px: float = 1.0,
) -> dict[str, Any]:
    force = np.asarray(requested_direction, dtype=np.float64)
    transform = np.asarray(canonical_to_image, dtype=np.float64)
    if force.shape != (2,) or not np.isfinite(force).all() or np.linalg.norm(force) <= 1e-12:
        raise ValueError("requested direction must be a finite nonzero 2-vector")
    if (
        transform.shape != (2, 2)
        or not np.isfinite(transform).all()
        or abs(np.linalg.det(transform)) < 1e-09
    ):
        raise ValueError("canonical-to-image transform must be finite and invertible")
    if (
        not np.isfinite(motion_signal_px)
        or motion_signal_px < 0
        or (not np.isfinite(motion_threshold_px))
        or (motion_threshold_px <= 0)
    ):
        raise ValueError("invalid motion signal/gate")
    result = {
        "schema": "uniaction-flower-inferred-force",
        "score": 0.0,
        "direction_cosine": None,
        "predicted_direction_xy": None,
        "motion_signal_px": float(motion_signal_px),
        "motion_threshold_px": motion_threshold_px,
        "reason": prediction["reason"],
        "applicable": False,
    }
    if not prediction["valid"]:
        return result
    predicted = np.asarray(prediction["direction_xy"], dtype=np.float64)
    if (
        predicted.shape != (2,)
        or not np.isfinite(predicted).all()
        or np.linalg.norm(predicted) <= 1e-12
    ):
        raise ValueError("predicted direction must be finite and nonzero")
    predicted = transform @ predicted
    predicted /= np.linalg.norm(predicted)
    result.update({"applicable": True, "predicted_direction_xy": predicted.tolist()})
    if motion_signal_px < motion_threshold_px:
        return {**result, "reason": "below_motion_gate"}
    cosine = float(predicted @ force / np.linalg.norm(force))
    return {
        **result,
        "reason": "ok",
        "score": float(np.clip(cosine, 0, 1)),
        "direction_cosine": cosine,
    }
