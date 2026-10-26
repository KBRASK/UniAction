from copy import deepcopy
import math

import torch

from .camera import JointOppositeCameraGuidance, opposite_camera_condition
from .inference import generate_sample


def counterfactual(sample):
    if sample.get("semantic_task") == "camera":
        return opposite_camera_condition(sample)
    result = deepcopy(sample)
    task = result.get("task")
    if task == "hand":
        for key in ("action", "action_time_mask", "action_point_time_mask"):
            result[key] = result[key][0:1].expand_as(result[key]).clone()
    elif task == "robot":
        uv, ids = result["action_token_uv"], result["action_token_id"]
        for joint in torch.unique(ids):
            mask = ids == joint
            uv[mask] = uv[torch.where(mask)[0][0]].clone()
    elif task == "force":
        uv = result["action_token_uv"]
        uv[1] = 2 * uv[0] - uv[1]
        if (uv < 0).any() or (uv > 1).any():
            raise ValueError(
                "Counterfactual force endpoint is outside the image; use separate guidance"
            )
    else:
        raise ValueError("Unsupported action for counterfactual guidance")
    return result


def validate(sample):
    if sample.get("task") not in ("hand", "robot", "force", "camera"):
        raise ValueError("Use an action from uniaction.conditions")
    for value in sample.values():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Action conditions must be finite")


@torch.no_grad()
def generate_counterfactual(config, pipe, adapter, sample, image, **kwargs):
    text, action = kwargs["text_cfg_scale"], kwargs["action_cfg_scale"]
    if not math.isclose(text, action) or text < 1:
        raise ValueError("Counterfactual guidance requires equal scales of at least one")
    validate(sample)
    shared = dict(kwargs, cfg_mode="shared_action", action_cfg_scale=1.0)
    if text == 1:
        return generate_sample(config, pipe, adapter, sample, image, **shared)
    with JointOppositeCameraGuidance(
        pipe,
        adapter,
        sample,
        scale=text,
        condition_validator=validate,
        opposite_builder=counterfactual,
    ) as guidance:
        frames = generate_sample(config, pipe, adapter, sample, image, **shared)
    if guidance.calls != 2 * config.evaluation.steps or guidance.forwards != guidance.calls:
        raise RuntimeError(
            "Counterfactual guidance forward count differs from the sampling schedule"
        )
    return frames
