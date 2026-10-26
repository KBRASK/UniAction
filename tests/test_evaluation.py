import importlib.util
import unittest

if any(importlib.util.find_spec(name) is None for name in ("cv2", "lpips", "scipy")):
    raise unittest.SkipTest("Optional dependencies are not installed")

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from uniaction.evaluation.api import ForceEvaluator, ReconstructionEvaluator, camera, trajectory
from uniaction.evaluation.camera_direction import score_direction
from uniaction.evaluation.camera_metrics import AXES, KEY_AXIS_SIGN
from uniaction.evaluation.pose import queries


class EvaluationTests(unittest.TestCase):
    def test_flower_default_bundle_loads_and_scores_without_external_assets(self):
        from uniaction.evaluation.flower_detector import FlowerDetector
        from uniaction.evaluation import api

        descriptor = Path(api.__file__).parents[1] / "assets/flower/model.json"
        explicit = FlowerDetector(descriptor)
        reference = explicit.view_bank.images[0]
        video = np.broadcast_to(reference, (81, *reference.shape))
        anchor = explicit.view_bank.anchors[0, 0]
        start = anchor / [832.0, 480.0]
        end = start + [0.05, 0.0]
        evaluator = ForceEvaluator()
        result = evaluator(video, start=start, end=end, kind="flower", reference_image=reference)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["applicable"])
        self.assertEqual(evaluator.flower.coefficients.shape, (25, 210, 2))
        expected = explicit.score_frames(
            video,
            reference_rgb=reference,
            start_xy=anchor,
            requested_direction=np.array([41.6, 0.0]),
        )["response"]
        self.assertEqual(result, expected)

    def test_camera_continuous_direction_and_matched_null(self):
        for key, (axis, sign) in KEY_AXIS_SIGN.items():
            reference = dict(reliable=True, late_axis_values=dict.fromkeys(AXES, 0.2))
            held = dict(reliable=True, late_axis_values=dict.fromkeys(AXES, 0.2))
            family = AXES[:3] if axis in AXES[:3] else AXES[3:]
            held["late_axis_values"][axis] += sign * 0.03
            held["late_axis_values"][next(a for a in family if a != axis)] += 0.04
            result = score_direction(held, reference, key)
            self.assertAlmostEqual(result["score"], 0.8)
            held["late_axis_values"][axis] -= sign * 0.06
            self.assertAlmostEqual(score_direction(held, reference, key)["score"], 0.2)

    def test_undefined_camera_is_not_reported_as_failure(self):
        reference = dict(reliable=True, late_axis_values=dict.fromkeys(AXES, 0.0))
        result = score_direction(reference, reference, "W")
        self.assertIsNone(result["score"])
        self.assertEqual(result["undefined_reason"], "zero_response")
        self.assertIsNone(score_direction(dict(reliable=False), reference, "W")["score"])

    def test_camera_anchors_absolute_pose_and_requires_late_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            p, q = Path(directory) / "pose.npz", Path(directory) / "null.npz"
            poses = np.repeat(np.eye(4)[None], 81, axis=0)
            poses[:, :3, 3] = [10.0, 20.0, 30.0]
            np.savez(q, data=poses, inds=np.arange(81))
            poses[:, 2, 3] += np.arange(81) / 100
            np.savez(p, data=poses, inds=np.arange(81))
            self.assertAlmostEqual(camera(p, q, "W")["score"], 1.0)
            np.savez(p, data=poses[:65], inds=np.arange(65))
            self.assertTrue(camera(p, q, "W")["reliable"])
            np.savez(p, data=poses[:61], inds=np.arange(61))
            self.assertIsNone(camera(p, q, "W")["score"])

    def test_pose_queries_never_use_future_robot_commands(self):
        uv = np.full((81, 12, 2), 0.5)
        valid = np.zeros((81, 12), bool)
        valid[[0, 40], 0] = True
        valid[[10, 50], 1] = True
        _, robot_times = queries(uv, valid, "robot")
        _, hand_times = queries(uv, valid, "hand")
        self.assertEqual(robot_times[0], 0)
        self.assertEqual(robot_times[1], -1)
        self.assertEqual(hand_times[1], 10)

    def test_invalid_initial_hand_seed_does_not_move_query_to_later_frame(self):
        uv = np.full((81, 12, 2), 0.5)
        valid = np.zeros((81, 12), bool)
        valid[[0, 10], 0] = True
        uv[0, 0, 0] = 1.0
        _, times = queries(uv, valid, "hand")
        self.assertEqual(times[0], -1)

    def test_trajectory_missing_points_and_still_predictions_do_not_gain_score(self):
        uv = np.full((81, 12, 2), 0.3)
        uv[:, :, 0] += np.arange(81)[:, None] / 1000
        mask = np.zeros((81, 12), bool)
        mask[:, :2] = True
        measured = uv * [832.0, 480.0]
        valid = mask.copy()
        self.assertAlmostEqual(
            trajectory(measured, valid, uv, mask, task="hand")["score_lower"], 1.0
        )
        valid[:, 1] = False
        result = trajectory(measured, valid, uv, mask, task="hand")
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertEqual(result["score"], result["score_lower"])
        self.assertAlmostEqual(result["conditional_score"], 1.0)
        self.assertAlmostEqual(result["score_upper"], 1.0)
        self.assertAlmostEqual(result["coverage"], 0.5)
        measured[:] = measured[0]
        self.assertEqual(trajectory(measured, mask, uv, mask, task="hand")["score_lower"], 0.0)

    def test_similarity_reports_only_lpips_and_excludes_first_frame(self):
        evaluator = object.__new__(ReconstructionEvaluator)
        evaluator.device, evaluator.model = torch.device("cpu"), None
        generated = np.zeros((81, 2, 2, 3), np.uint8)
        reference = generated.copy()
        generated[0] = 255
        with (
            patch("uniaction.evaluation.api.frames", side_effect=lambda value: value),
            patch("uniaction.evaluation.api._lpips_metric", return_value=0.0) as perceptual,
        ):
            result = evaluator(generated, reference)
        for actual in perceptual.call_args.args[1:3]:
            self.assertEqual(len(actual), 80)
            self.assertFalse(actual.any())
        self.assertEqual(result["first_frame"], 1)
        self.assertEqual(result["last_frame"], 80)
        self.assertEqual(result["lpips"], 0.0)
        self.assertEqual(
            set(result),
            {
                "lpips",
                "first_frame",
                "last_frame",
                "frames",
                "lpips_network",
                "lpips_version",
                "lpips_precision",
                "lpips_tf32",
            },
        )

    def test_similarity_cli_writes_lpips_receipt(self):
        from uniaction.cli.evaluate import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "metrics.json"
            video, reference = root / "video.mp4", root / "reference.mp4"
            video.touch()
            reference.touch()
            result = dict(lpips=0.25, first_frame=1, last_frame=80, frames=80)
            with (
                patch(
                    "sys.argv",
                    [
                        "uniaction-eval",
                        "similarity",
                        "--video",
                        str(video),
                        "--reference",
                        str(reference),
                        "--output",
                        str(output),
                    ],
                ),
                patch("uniaction.evaluation.api.ReconstructionEvaluator") as evaluator,
                patch("builtins.print"),
            ):
                evaluator.return_value.return_value = result
                main()
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["metric"], "similarity")
            self.assertEqual(receipt["result"], result)
            evaluator.return_value.assert_called_once_with(video, reference)

    def test_reconstruction_uses_full_precision_and_restores_caller_settings(self):
        evaluator = object.__new__(ReconstructionEvaluator)
        evaluator.device, evaluator.model = torch.device("cpu"), None
        video = np.zeros((81, 2, 2, 3), np.uint8)

        def measure(*args, **kwargs):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cudnn.benchmark)
            raise RuntimeError("measurement failed")

        with torch.backends.cudnn.flags(enabled=True, benchmark=True, allow_tf32=True):
            with (
                patch("uniaction.evaluation.api.frames", side_effect=lambda value: value),
                patch("uniaction.evaluation.api._lpips_metric", side_effect=measure),
            ):
                with self.assertRaisesRegex(RuntimeError, "measurement failed"):
                    evaluator(video, video)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cudnn.benchmark)


if __name__ == "__main__":
    unittest.main()
