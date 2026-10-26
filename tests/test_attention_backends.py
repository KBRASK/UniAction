from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from uniaction.backbones.wan.dit import DiTBlock
from uniaction.model.attention import (
    AttentionBackendRuntime,
    RuntimeAttentionModule,
    _fa4_plain_attention,
    inject_deferred_scalar_bias_gradient,
    inject_scalar_bias_gradient,
)
from uniaction.model.wan import attach_wan_attention_runtime


def _explicit_biased_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    video_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    key_bias = torch.cat(
        [
            bias.new_zeros(video_tokens),
            bias.new_ones(key.shape[-2] - video_tokens) * bias,
        ]
    )
    biased_scores = scores + key_bias.view(1, 1, 1, -1)
    probability = torch.softmax(biased_scores, dim=-1)
    output = torch.matmul(probability, value)
    return output, scores, biased_scores


class LearnableBiasGradientTests(unittest.TestCase):
    def test_fa4_scalar_gradient_formula_matches_explicit_attention(self) -> None:
        torch.manual_seed(1234)
        shape_q = (1, 2, 5, 8)
        shape_kv = (1, 2, 9, 8)
        video_tokens = 6
        query_ref = torch.randn(shape_q, dtype=torch.float64, requires_grad=True)
        key_ref = torch.randn(shape_kv, dtype=torch.float64, requires_grad=True)
        value_ref = torch.randn(shape_kv, dtype=torch.float64, requires_grad=True)
        bias_ref = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
        upstream = torch.randn(shape_q, dtype=torch.float64)
        output_ref, _, _ = _explicit_biased_attention(
            query_ref, key_ref, value_ref, bias_ref, video_tokens
        )
        gradients_ref = torch.autograd.grad(
            (output_ref * upstream).sum(),
            (query_ref, key_ref, value_ref, bias_ref),
        )

        query = query_ref.detach().clone().requires_grad_()
        key = key_ref.detach().clone().requires_grad_()
        value = value_ref.detach().clone().requires_grad_()
        bias = bias_ref.detach().clone().requires_grad_()
        output, scores, biased_scores = _explicit_biased_attention(
            query, key, value, bias.detach(), video_tokens
        )
        action_scores = scores[..., video_tokens:]
        action_probability = torch.softmax(action_scores, dim=-1)
        action_output = torch.matmul(action_probability, value[..., video_tokens:, :])
        action_lse = torch.logsumexp(action_scores, dim=-1)
        joint_lse = torch.logsumexp(biased_scores, dim=-1)
        action_mass = torch.exp(action_lse + bias.detach() - joint_lse)
        derivative = action_mass.unsqueeze(-1) * (action_output - output.detach())
        injected = inject_scalar_bias_gradient(output, bias, derivative.detach())
        gradients = torch.autograd.grad(
            (injected * upstream).sum(),
            (query, key, value, bias),
        )

        for actual, expected in zip(gradients, gradients_ref):
            torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    def test_deferred_fa4_scalar_gradient_matches_eager_formula(self) -> None:
        torch.manual_seed(1234)
        video_tokens = 6
        query_ref = torch.randn(1, 2, 5, 8, dtype=torch.float32, requires_grad=True)
        key_ref = torch.randn(1, 2, 9, 8, dtype=torch.float32, requires_grad=True)
        value_ref = torch.randn(1, 2, 9, 8, dtype=torch.float32, requires_grad=True)
        bias_ref = torch.tensor(1.3, dtype=torch.float32, requires_grad=True)
        upstream = torch.randn(1, 2, 5, 8, dtype=torch.float32)
        output_ref, _, _ = _explicit_biased_attention(
            query_ref, key_ref, value_ref, bias_ref, video_tokens
        )
        gradients_ref = torch.autograd.grad(
            (output_ref * upstream).sum(),
            (query_ref, key_ref, value_ref, bias_ref),
        )

        query = query_ref.detach().clone().requires_grad_()
        key = key_ref.detach().clone().requires_grad_()
        value = value_ref.detach().clone().requires_grad_()
        bias = bias_ref.detach().clone().requires_grad_()
        output, scores, biased_scores = _explicit_biased_attention(
            query, key, value, bias.detach(), video_tokens
        )
        action_scores = scores[..., video_tokens:]
        action_probability = torch.softmax(action_scores, dim=-1)
        action_output = torch.matmul(action_probability, value[..., video_tokens:, :])
        action_lse = torch.logsumexp(action_scores, dim=-1)
        joint_lse = torch.logsumexp(biased_scores, dim=-1)
        with patch(
            "flash_attn.cute.interface.flash_attn_func",
            return_value=(action_output.transpose(1, 2), action_lse),
        ):
            injected = inject_deferred_scalar_bias_gradient(
                output.transpose(1, 2),
                bias,
                query.transpose(1, 2),
                key[..., video_tokens:, :].transpose(1, 2),
                value[..., video_tokens:, :].transpose(1, 2),
                joint_lse,
                bias.detach(),
            )
            gradients = torch.autograd.grad(
                (injected * upstream.transpose(1, 2)).sum(),
                (query, key, value, bias),
            )

        for actual, expected in zip(gradients, gradients_ref):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


class BackendResolutionTests(unittest.TestCase):
    def test_requested_backend_controls_fallback_order(self) -> None:
        self.assertEqual(
            AttentionBackendRuntime(requested="auto")._candidates(),
            ("fa4", "fa3", "fa2", "legacy"),
        )
        self.assertEqual(
            AttentionBackendRuntime(requested="fa3")._candidates(),
            ("fa3", "fa2", "legacy"),
        )
        self.assertEqual(
            AttentionBackendRuntime(requested="fa4", allow_fallback=False)._candidates(),
            ("fa4",),
        )

    def test_resolution_records_failed_backend_and_actual_selection(self) -> None:
        runtime = AttentionBackendRuntime(requested="auto", allow_fallback=True)
        with patch.object(
            runtime,
            "_probe",
            side_effect=(RuntimeError("FA4 unavailable"), None),
        ):
            selected = runtime.resolve("cpu")
        self.assertEqual(selected, "fa3")
        summary = runtime.summary()
        self.assertEqual(summary["selected"], "fa3")
        self.assertEqual(summary["selected_id"], 3)
        self.assertEqual(
            [(attempt["backend"], attempt["status"]) for attempt in summary["attempts"]],
            [("fa4", "failed"), ("fa3", "selected")],
        )

    def test_resolution_falls_through_fa4_and_fa3_to_fa2(self) -> None:
        runtime = AttentionBackendRuntime(requested="auto", allow_fallback=True)
        with patch.object(
            runtime,
            "_probe",
            side_effect=(
                RuntimeError("FA4 unavailable"),
                RuntimeError("FA3 unavailable"),
                None,
            ),
        ):
            selected = runtime.resolve("cpu")
        self.assertEqual(selected, "fa2")
        self.assertEqual(
            [(attempt["backend"], attempt["status"]) for attempt in runtime.attempts],
            [("fa4", "failed"), ("fa3", "failed"), ("fa2", "selected")],
        )

    def test_resolution_falls_back_to_legacy_after_flash_probes_fail(self) -> None:
        runtime = AttentionBackendRuntime(requested="auto", allow_fallback=True)
        with patch.object(
            runtime,
            "_probe",
            side_effect=(
                RuntimeError("FA4 unavailable"),
                RuntimeError("FA3 unavailable"),
                RuntimeError("FA2 unavailable"),
            ),
        ) as probe:
            selected = runtime.resolve("cpu")
        self.assertEqual(selected, "legacy")
        self.assertEqual(probe.call_count, 3)
        self.assertEqual(
            [(attempt["backend"], attempt["status"]) for attempt in runtime.attempts],
            [
                ("fa4", "failed"),
                ("fa3", "failed"),
                ("fa2", "failed"),
                ("legacy", "selected"),
            ],
        )
        self.assertEqual(runtime.summary()["selected_id"], 0)

    def test_no_fallback_keeps_explicit_flash_backend_fail_closed(self) -> None:
        runtime = AttentionBackendRuntime(requested="fa4", allow_fallback=False)
        with patch.object(
            runtime,
            "_probe",
            side_effect=RuntimeError("FA4 unavailable"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "no requested attention backend passed its probe",
            ):
                runtime.resolve("cpu")
        self.assertIsNone(runtime.selected)
        self.assertEqual(
            [(attempt["backend"], attempt["status"]) for attempt in runtime.attempts],
            [("fa4", "failed")],
        )


class WholeWanAttentionRoutingTests(unittest.TestCase):
    def test_fa4_plain_attention_accepts_tuple_api_result(self) -> None:
        query = torch.randn(1, 5, 24)
        headed = query.reshape(1, 5, 3, 8)
        with patch(
            "flash_attn.cute.interface.flash_attn_func",
            return_value=(headed, torch.empty(0)),
        ):
            output = _fa4_plain_attention(
                query,
                query,
                query,
                num_heads=3,
            )
        torch.testing.assert_close(output, query)

    def test_self_text_and_image_cross_attention_share_one_runtime(self) -> None:
        dit = nn.Module()
        dit.blocks = nn.ModuleList(
            [DiTBlock(has_image_input=True, dim=24, num_heads=3, ffn_dim=48)]
        )
        state_keys_before = set(dit.state_dict())
        runtime = AttentionBackendRuntime(
            requested="legacy",
            allow_fallback=False,
            require_key_bias=False,
            selected="legacy",
            reference_count_learnable=False,
        )

        attach_wan_attention_runtime(dit, runtime)

        block = dit.blocks[0]
        self.assertIsInstance(block.self_attn.attn, RuntimeAttentionModule)
        self.assertIsInstance(block.cross_attn.attn, RuntimeAttentionModule)
        self.assertIs(block.self_attn.attn.runtime, runtime)
        self.assertIs(block.cross_attn.attn.runtime, runtime)
        self.assertSetEqual(set(dit.state_dict()), state_keys_before)

        query = torch.randn(1, 5, 24)
        block.self_attn.attn(query, query, query)
        context = torch.randn(1, 257 + 7, 24)
        output = block.cross_attn(query, context)
        self.assertEqual(tuple(output.shape), (1, 5, 24))
        self.assertEqual(runtime.plain_calls, 3)
        self.assertEqual(runtime.key_bias_calls, 0)
        self.assertEqual(runtime.summary()["coverage"], "wan_base_self_cross")
        self.assertFalse(runtime.summary()["learnable_reference_count"])

    def test_summary_separates_key_bias_from_frozen_reference_count(self) -> None:
        runtime = AttentionBackendRuntime(
            requested="legacy",
            allow_fallback=False,
            require_key_bias=True,
            selected="legacy",
            reference_count_learnable=False,
        )

        summary = runtime.summary()
        self.assertEqual(summary["summary_schema_version"], 2)
        self.assertTrue(summary["probe_requires_key_bias"])
        self.assertTrue(summary["action_key_bias_required"])
        self.assertTrue(summary["reference_count_gradient_supported"])
        self.assertFalse(summary["reference_count_learnable"])
        self.assertFalse(summary["learnable_reference_count"])

    def test_summary_reports_learnable_reference_count_when_configured(self) -> None:
        runtime = AttentionBackendRuntime(
            requested="legacy",
            allow_fallback=False,
            require_key_bias=True,
            selected="legacy",
            reference_count_learnable=True,
        )

        summary = runtime.summary()
        self.assertTrue(summary["action_key_bias_required"])
        self.assertTrue(summary["reference_count_learnable"])
        self.assertTrue(summary["learnable_reference_count"])

    def test_replacing_runtime_fails_closed(self) -> None:
        dit = nn.Module()
        dit.blocks = nn.ModuleList(
            [DiTBlock(has_image_input=False, dim=24, num_heads=3, ffn_dim=48)]
        )
        first = AttentionBackendRuntime(selected="legacy", requested="legacy")
        second = AttentionBackendRuntime(selected="legacy", requested="legacy")
        attach_wan_attention_runtime(dit, first)
        with self.assertRaisesRegex(RuntimeError, "different attention runtime"):
            attach_wan_attention_runtime(dit, second)


if __name__ == "__main__":
    unittest.main()
