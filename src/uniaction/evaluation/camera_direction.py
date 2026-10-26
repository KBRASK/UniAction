from __future__ import annotations
import math
from collections.abc import Mapping
from uniaction.evaluation.camera_metrics import AXES, KEY_AXIS_SIGN

SPECIFICATION = "camera-matched-null-affine-cosine-v1"


def score_direction(held_pose: Mapping, null_pose: Mapping, key: str) -> dict:
    if key not in KEY_AXIS_SIGN:
        raise ValueError("expected one held WASD/IJKL key, not null_action")
    axis, sign = KEY_AXIS_SIGN[key]
    family = "translation" if axis in AXES[:3] else "rotation"
    axes = AXES[:3] if family == "translation" else AXES[3:]
    reliable = held_pose.get("reliable") is True and null_pose.get("reliable") is True
    result = dict(
        specification=SPECIFICATION,
        key=key,
        axis=axis,
        family=family,
        reliable=reliable,
        score=None,
        cosine=None,
        direction_defined=False,
        undefined_reason="unreliable_pose" if not reliable else None,
        axis_delta_from_null=None,
        response_norm=None,
        requested_signed_response=None,
    )
    if not reliable:
        return result
    try:
        held = [float(held_pose["late_axis_values"][a]) for a in AXES]
        null = [float(null_pose["late_axis_values"][a]) for a in AXES]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("reliable pose requires all six finite late-axis values") from error
    if not all((math.isfinite(x) for x in held + null)):
        raise ValueError("non-finite reliable pose summary")
    delta = dict(zip(AXES, (a - b for a, b in zip(held, null))))
    if not all((math.isfinite(x) for x in delta.values())):
        raise ValueError("non-finite matched-null response")
    vector = [delta[a] for a in axes]
    norm = math.hypot(*vector)
    if not math.isfinite(norm):
        raise ValueError("matched-null response norm overflow")
    signed = sign * delta[axis]
    scale = max((abs(x) for x in vector))
    result.update(axis_delta_from_null=delta, response_norm=norm, requested_signed_response=signed)
    if not scale:
        result["undefined_reason"] = "zero_response"
        return result
    cosine = signed / scale / math.hypot(*(x / scale for x in vector))
    cosine = max(-1.0, min(1.0, cosine))
    result.update(score=(1.0 + cosine) / 2.0, cosine=cosine, direction_defined=True)
    return result
