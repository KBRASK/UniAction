from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from uniaction.inference import (
    _base_model_prediction,
    _generate_explicit_cfg,
    combine_cfg_predictions,
    generate_sample,
)
from uniaction.model.lora import inject_lora


class CFGGuidanceTests(unittest.TestCase):
    def test_joint_cfg_uses_true_empty_text_no_action_branch(self) -> None:
        action = torch.tensor([10.0])
        negative = torch.tensor([1.0])

        prediction = combine_cfg_predictions(
            action,
            mode="joint",
            text_scale=3.0,
            action_scale=3.0,
            base_negative=negative,
        )

        torch.testing.assert_close(prediction, torch.tensor([28.0]))

    def test_separate_cfg_composes_text_and_action_deltas(self) -> None:
        prediction = combine_cfg_predictions(
            torch.tensor([10.0]),
            mode="separate",
            text_scale=3.0,
            action_scale=2.0,
            base_positive=torch.tensor([4.0]),
            base_negative=torch.tensor([1.0]),
        )

        torch.testing.assert_close(prediction, torch.tensor([22.0]))

    def test_separate_unit_scales_collapse_to_action_prediction(self) -> None:
        action = torch.randn(2, 3)
        prediction = combine_cfg_predictions(
            action,
            mode="separate",
            text_scale=1.0,
            action_scale=1.0,
            base_positive=torch.randn(2, 3),
        )
        torch.testing.assert_close(prediction, action)

    def test_equal_separate_scales_eliminate_text_only_branch_exactly(self) -> None:
        action = torch.tensor([10.0, -2.0])
        empty = torch.tensor([1.0, 4.0])

        for scale in (3.0, 5.0):
            with self.subTest(scale=scale):
                reduced = combine_cfg_predictions(
                    action,
                    mode="separate",
                    text_scale=scale,
                    action_scale=scale,
                    base_negative=empty,
                )

                text_only = torch.tensor([7.0, -9.0])
                unreduced = (
                    text_only + scale * (action - text_only) + (scale - 1.0) * (text_only - empty)
                )
                torch.testing.assert_close(reduced, unreduced)

    def test_equal_separate_scales_execute_two_denoiser_branches_per_step(self) -> None:
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                steps=2,
                sigma_shift=5.0,
                solver="euler",
            ),
            model=SimpleNamespace(
                height=4,
                width=4,
                num_frames=3,
                switch_dit_boundary=0.0,
            ),
        )

        class Scheduler:
            timesteps = (torch.tensor(900.0), torch.tensor(100.0))

            def set_timesteps(self, *_args, **_kwargs):
                return None

            def step(self, prediction, _timestep, _latents):
                return prediction

        class VAE:
            @staticmethod
            def decode(latents, **_kwargs):
                return latents

        class Pipe:
            dit = object()
            dit2 = None
            vae = VAE()
            text_encoder = object()
            tokenizer = object()
            scheduler = Scheduler()
            units = (object(),)
            torch_dtype = torch.float32
            device = torch.device("cpu")

            def __init__(self):
                self.action_calls = 0

            @staticmethod
            def unit_runner(_unit, _pipe, shared, positive, negative):
                return {**shared, "latents": torch.zeros(1)}, positive, negative

            def model_fn(self, **_kwargs):
                self.action_calls += 1
                return torch.tensor([10.0])

            @staticmethod
            def vae_output_to_video(_video):
                return ["generated"]

        pipe = Pipe()
        with patch(
            "uniaction.inference._base_model_prediction",
            return_value=torch.tensor([1.0]),
        ) as base_prediction:
            frames = _generate_explicit_cfg(
                config,
                pipe,
                object(),
                prompt="test",
                negative_prompt="negative",
                seed=1234,
                mode="separate",
                text_scale=3.0,
                action_scale=3.0,
            )

        self.assertEqual(frames, ["generated"])
        self.assertEqual(pipe.action_calls, 2)
        self.assertEqual(base_prediction.call_count, 2)

    def test_separate_unit_scales_execute_one_pipeline_branch(self) -> None:
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                cfg_scale=1.0,
                seed=1234,
                steps=20,
                sigma_shift=5.0,
                solver="euler",
            ),
            model=SimpleNamespace(
                height=480,
                width=832,
                num_frames=81,
                switch_dit_boundary=0.875,
            ),
        )

        class Pipe:
            device = torch.device("cpu")
            torch_dtype = torch.float32

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return ["generated"]

        pipe = Pipe()
        dit = object()
        with (
            patch("uniaction.inference.iter_wan_dits", return_value=(("dit", dit),)),
            patch("uniaction.inference.stage_sample_condition"),
            patch("uniaction.inference.clear_action_condition"),
            patch("uniaction.inference._generate_explicit_cfg") as explicit,
        ):
            frames = generate_sample(
                config,
                pipe,
                object(),
                {"task": "force"},
                object(),
                prompt="test",
                cfg_mode="separate",
                text_cfg_scale=1.0,
                action_cfg_scale=1.0,
            )

        self.assertEqual(frames, ["generated"])
        self.assertEqual(len(pipe.calls), 1)
        self.assertEqual(pipe.calls[0]["cfg_scale"], 1.0)
        explicit.assert_not_called()

    def test_joint_cfg_rejects_mismatched_scales(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal text and action scales"):
            combine_cfg_predictions(
                torch.ones(1),
                mode="joint",
                text_scale=3.0,
                action_scale=2.0,
                base_negative=torch.zeros(1),
            )

    def test_base_prediction_suspends_action_and_lora_then_restores_both(self) -> None:
        class TinyDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q = nn.Linear(2, 2, bias=False)

        dit = inject_lora(
            TinyDiT(),
            target_modules=("q",),
            rank=1,
            alpha=1,
            dtype=torch.float32,
        )
        with torch.no_grad():
            dit.q.base_layer.weight.copy_(torch.eye(2))
            dit.q.lora_A["default"].weight.fill_(1.0)
            dit.q.lora_B["default"].weight.fill_(2.0)
        original_state = SimpleNamespace(tokens=torch.ones(1))
        dit._action_state = original_state
        requires_grad = {
            name: parameter.requires_grad for name, parameter in dit.named_parameters()
        }

        class Pipe:
            @staticmethod
            def model_fn(*, dit, value):
                action_marker = 100.0 if dit._action_state.tokens is not None else 0.0
                return dit.q(value) + action_marker

        value = torch.tensor([[1.0, 3.0]])
        enabled = Pipe.model_fn(dit=dit, value=value)
        base = _base_model_prediction(Pipe(), dit, {"value": value})

        self.assertIs(dit._action_state, original_state)
        self.assertFalse(dit.q._disable_adapters)
        self.assertEqual(
            requires_grad,
            {name: parameter.requires_grad for name, parameter in dit.named_parameters()},
        )
        torch.testing.assert_close(base, value)
        self.assertFalse(torch.equal(enabled, base))


if __name__ == "__main__":
    unittest.main()
