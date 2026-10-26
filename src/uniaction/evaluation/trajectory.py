from __future__ import annotations
import numpy as np


def _result(values, observed, eligible):
    counts = eligible.sum(axis=0)
    points = counts > 0
    if not points.any():
        return dict(
            score_lower=None,
            score_upper=None,
            conditional_score=None,
            coverage=0.0,
            eligible_point_times=0,
            eligible_points=0,
            reason="no_eligible_command",
        )
    n = counts[points]
    coverage = float(np.mean(observed[:, points].sum(axis=0) / n))
    lower = float(np.mean(values[:, points].sum(axis=0) / n))
    return dict(
        score_lower=lower,
        score_upper=min(1.0, lower + 1.0 - coverage),
        conditional_score=lower / coverage if coverage else None,
        coverage=coverage,
        eligible_point_times=int(counts.sum()),
        eligible_points=int(points.sum()),
        reason="ok" if coverage == 1 else "partial_observation",
    )


def trajectory_score(
    measured_px,
    measured_valid,
    command_px,
    command_valid,
    *,
    static_command=False,
    moving_command_min_px=4.0,
    measured_motion_min_px=2.0,
    static_tolerance_px=4.0,
):
    m, c = (np.asarray(v, dtype=float) for v in (measured_px, command_px))
    mv, cv = (np.asarray(v, dtype=bool) for v in (measured_valid, command_valid))
    if (
        c.ndim != 3
        or c.shape[-1] != 2
        or len(c) < 2
        or (m.shape != c.shape)
        or (mv.shape != c.shape[:2])
        or (cv.shape != mv.shape)
    ):
        raise ValueError("trajectory shapes must agree: T,K,2 with T,K masks")
    gates = np.asarray([moving_command_min_px, measured_motion_min_px, static_tolerance_px])
    if not np.isfinite(gates).all() or (gates <= 0).any():
        raise ValueError("gates must be finite and positive")
    cv = cv & np.isfinite(c).all(axis=-1)
    mv = mv & np.isfinite(m).all(axis=-1)
    eligible, observed = (np.zeros(cv.shape, bool), np.zeros(cv.shape, bool))
    values = np.zeros(cv.shape, float)
    for point in range(c.shape[1]):
        frames = np.flatnonzero(cv[:, point])
        if len(frames) < 2:
            continue
        anchor, frames = (int(frames[0]), frames[1:])
        dc = c[frames, point] - c[anchor, point]
        cn = np.linalg.norm(dc, axis=-1)
        if static_command and np.any(cn > static_tolerance_px):
            raise ValueError("static_command requested for a moving target")
        use = np.ones(len(frames), bool) if static_command else cn >= moving_command_min_px
        frames, dc = (frames[use], dc[use])
        eligible[frames, point] = True
        if not mv[anchor, point]:
            continue
        good = mv[frames, point]
        f, dc = (frames[good], dc[good])
        observed[f, point] = True
        dm = m[f, point] - m[anchor, point]
        mn = np.linalg.norm(dm, axis=-1)
        if static_command:
            score = np.clip(1.0 - mn / static_tolerance_px, 0.0, 1.0)
        else:
            denominator = np.square(dm).sum(axis=-1) + np.square(dc).sum(axis=-1)
            score = np.clip(2.0 * (dm * dc).sum(axis=-1) / np.maximum(denominator, 1e-12), 0.0, 1.0)
            score[mn < measured_motion_min_px] = 0.0
        values[f, point] = score
    return _result(values, observed, eligible)
