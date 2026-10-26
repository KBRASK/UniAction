from __future__ import annotations
import numpy as np


def predict_ridge_direction(model, query, valid_mask):
    q, mask = (np.asarray(query, dtype=float), np.asarray(valid_mask, dtype=bool))
    if q.shape != model["scale"].shape or mask.shape != q.shape or (not np.isfinite(q).all()):
        raise ValueError("invalid ridge query features")
    if np.count_nonzero(mask) < 8:
        return {"valid": False, "reason": "no_common_observable_flow", "direction_xy": None}
    predicted = np.where(mask, q / model["scale"], 0.0) @ model["coefficients"]
    magnitude = float(np.linalg.norm(predicted))
    if magnitude <= 1e-09:
        return {"valid": False, "reason": "no_inferred_force", "direction_xy": None}
    return {
        "valid": True,
        "reason": "ok",
        "direction_xy": (predicted / magnitude).tolist(),
        "inferred_force_magnitude": magnitude,
    }
