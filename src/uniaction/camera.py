from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import math

import numpy as np
import torch

from uniaction.data.camera_keys import (
    CAMERA_KEYS,
    KEY_TO_ID,
    OPPOSITE_KEY,
    SCREEN_SHARED_UV,
    key_tokens,
)
from uniaction.model.action import stage_sample_condition


SPECIFICATION = "camera-screen-shared-native-intervals-v1"
JOINT_GUIDANCE_SPECIFICATION = "joint_positive_text_action_vs_negative_text_opposite_v1"


def camera_joint_specification(text_scale, action_scale):

    if any(not math.isfinite(value) or value <= 1 for value in (text_scale, action_scale)):
        raise ValueError("joint camera guidance requires finite scales >1")
    if text_scale != action_scale:
        raise ValueError("joint camera guidance requires equal text and action scales")
    return 2, JOINT_GUIDANCE_SPECIFICATION


def held_camera_condition(sample, key):

    if key not in ("null_action", *CAMERA_KEYS):
        raise ValueError("camera evaluation accepts null_action or one WASD/IJKL key")
    times = np.asarray(sample["camera_event_times"], dtype=np.float32)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or bool((times < 0).any())
        or bool((np.diff(times) <= 0).any())
    ):
        raise ValueError("camera needs a finite, strictly increasing native event grid")
    result = {
        **sample,
        **key_tokens([() if key == "null_action" else (key,)] * len(times), times),
        "semantic_task": "camera",
        "task": "camera",
        "camera_key_layout": "screen_shared",
        "camera_condition": key,
        "action_token_vocab_size": 16,
    }
    validate_camera_condition(result)
    return result


def validate_camera_condition(sample):
    if "action" in sample:
        raise ValueError("camera condition cannot also contain a dense action branch")
    if (
        sample.get("semantic_task") != "camera"
        or sample.get("task") != "camera"
        or sample.get("camera_key_layout") != "screen_shared"
    ):
        raise ValueError("expected explicit camera semantics and the shared sparse checkpoint slot")
    key = sample.get("camera_condition")
    if key not in ("null_action", *CAMERA_KEYS):
        raise ValueError("only held single-key camera conditions are supported")
    uv = torch.as_tensor(sample["action_token_uv"]).detach().cpu()
    times = torch.as_tensor(sample["action_token_time"]).detach().cpu()
    ids = torch.as_tensor(sample["action_token_id"]).detach().cpu()
    if (
        ids.ndim != 1
        or times.shape != ids.shape
        or uv.shape != (len(ids), 2)
        or ids.dtype != torch.int64
        or not bool(torch.isfinite(uv).all())
        or not bool(torch.isfinite(times).all())
    ):
        raise ValueError("invalid camera sparse tensor inventory/values")
    if key == "null_action":
        if len(ids):
            raise ValueError("null camera condition must have zero tokens")
        return
    if (
        not len(ids)
        or not bool((ids == KEY_TO_ID[key]).all())
        or bool((times < 0).any())
        or bool((times[1:] <= times[:-1]).any())
    ):
        raise ValueError("held camera IDs/times disagree with the requested key")
    expected_uv = torch.from_numpy(SCREEN_SHARED_UV[ids.numpy()].copy())
    if not torch.equal(uv.float(), expected_uv):
        raise ValueError("camera key UV is not the pinned screen_shared layout")
    if not torch.equal(
        times.float(), torch.as_tensor(sample["camera_event_times"], dtype=torch.float32)
    ):
        raise ValueError("camera token times must preserve every native event midpoint")
    if "video" in sample and bool((times > len(sample["video"]) - 1).any()):
        raise ValueError("camera events exceed the selected RGB view")


def opposite_camera_condition(sample):

    validate_camera_condition(sample)
    key = sample["camera_condition"]
    if key == "null_action":
        return None
    negative_key = OPPOSITE_KEY[key]
    ids = sample["action_token_id"]
    uv = sample["action_token_uv"]
    return {
        **sample,
        "camera_condition": negative_key,
        "action_token_id": torch.full_like(ids, KEY_TO_ID[negative_key]),
        "action_token_time": sample["action_token_time"].clone(),
        "action_token_uv": torch.as_tensor(
            SCREEN_SHARED_UV[KEY_TO_ID[negative_key]].copy(), device=uv.device, dtype=uv.dtype
        )
        .expand_as(uv)
        .clone(),
    }


class JointOppositeCameraGuidance:
    def __init__(
        self,
        pipe,
        adapter,
        sample,
        *,
        scale,
        condition_validator=validate_camera_condition,
        opposite_builder=opposite_camera_condition,
    ):
        if not math.isfinite(scale) or scale < 1:
            raise ValueError("opposite camera scale must be finite and >=1")
        condition_validator(sample)
        self.pipe, self.adapter, self.scale = pipe, adapter, float(scale)
        fields = ("action_token_uv", "action_token_time", "action_token_id", "action_token_value")
        self.positive = {**sample, **{k: sample[k].detach().clone() for k in fields if k in sample}}
        self.negative = opposite_builder(self.positive)
        if self.negative is None:
            raise ValueError("null camera uses the Stage-1 base, not opposite guidance")
        self.original = None
        self.calls = self.forwards = self.guided_calls = self.base_calls = 0

    def _stage(self, dit, sample):
        stage_sample_condition(
            dit, self.adapter, sample, device=self.pipe.device, dtype=self.pipe.torch_dtype
        )

    def __exit__(self, *_exc):
        if self.original is not None:
            self.pipe.model_fn, self.original = self.original, None

    def __enter__(self):
        if self.original is not None:
            raise RuntimeError("camera guidance cannot be reentered")
        self.original = self.pipe.model_fn
        original = self.original

        def guided(*, dit, **inputs):
            positive_text = self.calls % 2 == 0
            self.calls += 1
            try:
                self._stage(dit, self.positive if positive_text else self.negative)
                self.forwards += 1
                return original(dit=dit, **inputs)
            finally:
                self._stage(dit, self.positive)

        self.pipe.model_fn = guided
        return self


def generate_joint_camera_sample(config, pipe, adapter, sample, first_frame, **kwargs):

    from uniaction.inference import generate_sample

    text, action = kwargs["text_cfg_scale"], kwargs["action_cfg_scale"]
    branches, name = camera_joint_specification(text, action)
    with JointOppositeCameraGuidance(pipe, adapter, sample, scale=action) as guidance:
        frames = generate_sample(
            config,
            pipe,
            adapter,
            sample,
            first_frame,
            **dict(kwargs, cfg_mode="shared_action", text_cfg_scale=text, action_cfg_scale=1.0),
        )
    if guidance.calls != config.evaluation.steps * branches or guidance.forwards != guidance.calls:
        raise RuntimeError("joint camera denoiser branch verification failed")
    sample["_camera_joint_runtime"] = dict(
        specification=name,
        text_scale=float(text),
        action_scale=float(action),
        calls=guidance.calls,
        forwards=guidance.forwards,
        base_calls=0,
        action_reads_video=bool(adapter.action_reads_video),
    )
    return frames


@torch.no_grad()
def generate_camera_sample(
    config,
    pipe,
    adapter,
    sample,
    first_frame,
    *,
    condition_validator=validate_camera_condition,
    opposite_builder=opposite_camera_condition,
    **kwargs,
):

    from uniaction.inference import generate_i2v_sample, generate_sample
    from uniaction.model.action import suspend_action_condition
    from uniaction.model.lora import disable_lora_adapters
    from uniaction.model.wan import iter_wan_dits

    condition_validator(sample)
    mode = kwargs.get("cfg_mode", "separate")
    text_scale = kwargs.get("text_cfg_scale")
    text_scale = config.evaluation.cfg_scale if text_scale is None else text_scale
    action_scale = kwargs.get("action_cfg_scale")
    action_scale = 1.0 if action_scale is None else action_scale
    if mode == "camera_joint":
        return generate_joint_camera_sample(
            config,
            pipe,
            adapter,
            sample,
            first_frame,
            **dict(kwargs, text_cfg_scale=text_scale, action_cfg_scale=action_scale),
        )
    if mode not in ("separate", "shared_action", "joint"):
        raise ValueError("unsupported camera CFG mode")
    if any(not math.isfinite(value) or value <= 0 for value in (text_scale, action_scale)):
        raise ValueError("camera CFG scales must be finite and positive")
    if sample["camera_condition"] == "null_action":
        base_config = replace(config, evaluation=replace(config.evaluation, cfg_scale=text_scale))
        with ExitStack() as context:
            for _, dit in iter_wan_dits(pipe):
                context.enter_context(suspend_action_condition(dit))
                context.enter_context(disable_lora_adapters(dit))
            return generate_i2v_sample(
                base_config,
                pipe,
                first_frame,
                prompt=kwargs["prompt"],
                negative_prompt=kwargs.get("negative_prompt", ""),
                seed=kwargs.get("seed"),
            )
    return generate_sample(config, pipe, adapter, sample, first_frame, **kwargs)
