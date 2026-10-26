from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from uniaction.backbones.wan.pipeline import WanPipeline
from uniaction.data.camera_keys import KEY_TO_ID
from uniaction.camera import (
    JOINT_GUIDANCE_SPECIFICATION,
    generate_camera_sample,
    held_camera_condition,
)
from uniaction.model.action import _ActionState


class TinyAdapter:
    action_reads_video = True

    def encode_sample(self, sample, *, device, dtype):
        ids = sample["action_token_id"]
        return (
            ids[None, :, None].to(device=device, dtype=dtype),
            sample["action_token_uv"][None].to(device=device),
            torch.ones((1, len(ids)), device=device, dtype=torch.bool),
            sample["action_token_time"][None].to(device=device),
        )


class JointCameraPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            evaluation=SimpleNamespace(
                cfg_scale=5.0,
                steps=2,
                seed=1234,
                sigma_shift=5.0,
                solver="euler",
            ),
            model=SimpleNamespace(
                height=16,
                width=16,
                num_frames=5,
                switch_dit_boundary=0.875,
            ),
        )
        self.adapter = TinyAdapter()
        self.sample = held_camera_condition({"camera_event_times": np.arange(4) + 0.5}, "W")

    def pipeline(self, *, dual=False, fail_context=None):
        pipe = WanPipeline(device="cpu", torch_dtype=torch.float32)
        pipe.dit = SimpleNamespace(_action_adapter_attached=True, _action_state=_ActionState())
        if dual:
            pipe.dit2 = SimpleNamespace(_action_adapter_attached=True, _action_state=_ActionState())
        pipe.vae = SimpleNamespace(decode=lambda latents, **kwargs: latents)
        pipe.text_encoder = pipe.tokenizer = object()
        pipe.units = [object()]

        def unit_runner(unit, pipeline, shared, positive, negative):
            self.assertEqual(positive, {"prompt": "scene"})
            self.assertEqual(negative, {"negative_prompt": "negative"})
            return (
                {**shared, "latents": torch.zeros(1, 1, 1, 1, 1)},
                {"context": "pos"},
                {"context": "neg"},
            )

        pipe.unit_runner = unit_runner
        pipe.vae_output_to_video = lambda video: video
        calls, scheduled = [], []

        def model_fn(*, dit, latents, timestep, context, **kwargs):
            key = int(dit._action_state.tokens[0, 0, 0])
            calls.append((context, key, latents, timestep, dit))
            if context == fail_context:
                raise RuntimeError(f"injected {context} branch failure")
            value = {("pos", KEY_TO_ID["W"]): 11.0, ("neg", KEY_TO_ID["S"]): 1.0}[context, key]
            return torch.full_like(latents, value)

        original_step = pipe.scheduler.step

        def step(prediction, *args, **kwargs):
            scheduled.append(prediction.clone())
            return original_step(prediction, *args, **kwargs)

        pipe.model_fn, pipe.scheduler.step = model_fn, step
        return pipe, calls, scheduled, model_fn, step

    def generate(self, pipe, **scales):
        return generate_camera_sample(
            self.config,
            pipe,
            self.adapter,
            self.sample,
            None,
            prompt="scene",
            negative_prompt="negative",
            cfg_mode="camera_joint",
            **scales,
        )

    def assert_restored(self, pipe, model_fn, step):
        self.assertIs(pipe.model_fn, model_fn)
        self.assertIs(pipe.scheduler.step, step)
        for dit in (pipe.dit, pipe.dit2):
            if dit is not None:
                self.assertEqual(dit._action_state, _ActionState())
        self.assertTrue(self.adapter.action_reads_video)

    def test_real_loop_combines_once_at_same_latent_and_stages_active_expert(self):
        for dual in (False, True):
            with self.subTest(dual=dual):
                pipe, calls, scheduled, model_fn, step = self.pipeline(dual=dual)
                self.generate(pipe, text_cfg_scale=5.0, action_cfg_scale=5.0)
                self.assertEqual([(c[0], c[1]) for c in calls], [("pos", 0), ("neg", 2)] * 2)
                self.assertEqual([float(value.item()) for value in scheduled], [51.0, 51.0])
                for index in (0, 2):
                    positive, negative = calls[index : index + 2]
                    self.assertIs(positive[2], negative[2])
                    self.assertIs(positive[3], negative[3])
                    self.assertIs(positive[4], negative[4])
                self.assertIs(calls[0][4], pipe.dit)
                self.assertIs(calls[2][4], pipe.dit2 if dual else pipe.dit)
                self.assertEqual(self.sample["_camera_joint_runtime"]["forwards"], 4)
                self.assert_restored(pipe, model_fn, step)

    def test_omitted_or_none_text_scale_uses_config_default(self):
        for scales in (
            {"action_cfg_scale": 5.0},
            {"text_cfg_scale": None, "action_cfg_scale": 5.0},
        ):
            with self.subTest(scales=scales):
                pipe, calls, scheduled, model_fn, step = self.pipeline()
                self.generate(pipe, **scales)
                self.assertEqual([float(value.item()) for value in scheduled], [51.0, 51.0])
                self.assertEqual(self.sample["_camera_joint_runtime"]["text_scale"], 5.0)
                self.assert_restored(pipe, model_fn, step)

    def test_default_action_scale_fails_specification_before_sampling(self):
        for scales in ({"text_cfg_scale": 5.0}, {"text_cfg_scale": 5.0, "action_cfg_scale": None}):
            with self.subTest(scales=scales):
                pipe, calls, scheduled, model_fn, step = self.pipeline()
                with self.assertRaisesRegex(ValueError, "finite scales >1"):
                    self.generate(pipe, **scales)
                self.assertEqual(calls, [])
                self.assert_restored(pipe, model_fn, step)

    def test_either_branch_failure_restores_callable_and_clears_all_experts(self):
        for fail_context in ("pos", "neg"):
            with self.subTest(fail_context=fail_context):
                pipe, calls, scheduled, model_fn, step = self.pipeline(
                    dual=True, fail_context=fail_context
                )
                with self.assertRaisesRegex(
                    RuntimeError, f"injected {fail_context} branch failure"
                ):
                    self.generate(pipe, text_cfg_scale=5.0, action_cfg_scale=5.0)
                self.assertEqual(scheduled, [])
                self.assert_restored(pipe, model_fn, step)

    def test_null_requires_separate_stage1_reference(self):
        pipe, calls, scheduled, model_fn, step = self.pipeline()
        self.sample = held_camera_condition(self.sample, "null_action")
        with self.assertRaisesRegex(ValueError, "null camera uses the Stage-1 base"):
            self.generate(pipe, text_cfg_scale=5.0, action_cfg_scale=5.0)
        self.assertEqual(calls, [])
        self.assert_restored(pipe, model_fn, step)

    def test_missing_forwards_cannot_produce_runtime_receipt(self):
        pipe, _, _, model_fn, step = self.pipeline()
        with patch("uniaction.inference.generate_sample", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "branch verification"):
                self.generate(pipe, text_cfg_scale=5.0, action_cfg_scale=5.0)
        self.assertNotIn("_camera_joint_runtime", self.sample)
        self.assert_restored(pipe, model_fn, step)

    def test_joint_runtime_records_active_guidance(self):
        pipe, _, _, _, _ = self.pipeline()
        self.generate(pipe, text_cfg_scale=5.0, action_cfg_scale=5.0)
        runtime = self.sample["_camera_joint_runtime"]
        self.assertEqual(runtime["specification"], JOINT_GUIDANCE_SPECIFICATION)
        self.assertTrue(runtime["action_reads_video"])


if __name__ == "__main__":
    unittest.main()
