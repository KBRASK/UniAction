import numpy as np
from .trajectory import _result, trajectory_score


def displacement_pairs(measured, measured_valid, command, command_valid, gate=4.0):
    m, c = (np.asarray(x, float) for x in (measured, command))
    mv, cv = (np.asarray(x, bool).copy() for x in (measured_valid, command_valid))
    if (
        c.ndim != 3
        or c.shape[-1] != 2
        or len(c) < 2
        or (m.shape != c.shape)
        or (mv.shape != c.shape[:2])
        or (cv.shape != mv.shape)
        or (not np.isfinite(gate))
        or (gate <= 0)
    ):
        raise ValueError("compatible T,K,2 arrays and positive target gate required")
    cv &= np.isfinite(c).all(-1)
    mv &= np.isfinite(m).all(-1)
    eligible = np.zeros(cv.shape, bool)
    observed = np.zeros(cv.shape, bool)
    error = np.full(cv.shape, np.nan)
    extent = np.zeros(cv.shape)
    motion = np.zeros(cv.shape)
    for point in range(c.shape[1]):
        times = np.flatnonzero(cv[:, point])
        if len(times) < 2:
            continue
        anchor = times[0]
        times = times[1:]
        dc = c[times, point] - c[anchor, point]
        r = np.linalg.norm(dc, axis=-1)
        keep = r >= gate
        times, dc, r = (times[keep], dc[keep], r[keep])
        eligible[times, point] = True
        extent[times, point] = r
        if not mv[anchor, point]:
            continue
        good = mv[times, point]
        times, dc = (times[good], dc[good])
        observed[times, point] = True
        dm = m[times, point] - m[anchor, point]
        error[times, point] = np.linalg.norm(dm - dc, axis=-1)
        motion[times, point] = np.linalg.norm(dm, axis=-1)
    return dict(error=error, extent=extent, motion=motion, eligible=eligible, observed=observed)


def point_time_weights(eligible):
    eligible = np.asarray(eligible, bool)
    counts = eligible.sum(0)
    points = counts > 0
    weights = np.zeros(eligible.shape, float)
    if points.any():
        weights[:, points] = eligible[:, points] / counts[points] / points.sum()
    return weights


def score(
    measured,
    measured_valid,
    command,
    command_valid,
    *,
    tolerance_px,
    cap_fraction=0.25,
    command_min=4.0,
    measured_min=2.0,
    static_command=False,
):
    if (
        not np.isfinite(tolerance_px)
        or tolerance_px < 0
        or (not 0 <= cap_fraction < 1)
        or (not np.isfinite(measured_min))
        or (measured_min <= 0)
    ):
        raise ValueError("finite nonnegative tolerance, cap<1 and positive motion gate required")
    if static_command:
        result = trajectory_score(
            measured,
            measured_valid,
            command,
            command_valid,
            static_command=True,
            moving_command_min_px=command_min,
            measured_motion_min_px=measured_min,
            static_tolerance_px=4.0,
        )
        return dict(result, tolerance_cap_fraction=None, static=True)
    pair = displacement_pairs(measured, measured_valid, command, command_valid, command_min)
    obs = pair["observed"]
    r = pair["extent"]
    e = pair["error"]
    tau = np.minimum(tolerance_px, cap_fraction * r)
    values = np.zeros(r.shape, float)
    values[obs] = np.clip((r[obs] - e[obs]) / (r[obs] - tau[obs]), 0, 1) * (
        pair["motion"][obs] >= measured_min
    )
    weights = point_time_weights(pair["eligible"])
    mass = weights[obs].sum()
    cap_mass = weights[obs & (tolerance_px > cap_fraction * r)].sum()
    result = _result(values, obs, pair["eligible"])
    return dict(
        result, tolerance_cap_fraction=float(cap_mass / mass) if mass else None, static=False
    )
