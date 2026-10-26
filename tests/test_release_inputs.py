from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from uniaction import conditions
from uniaction.checkpoints import read_release, release_metadata, verify_checkpoints
from uniaction.config import make_config
from uniaction.model.action import ActionAdapter, UnifiedActionConfig


class ReleaseInputsTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ActionAdapter(
            UnifiedActionConfig(
                num_frames=81, dit_dim=64, num_dit_blocks=2, torch_dtype=torch.float32
            )
        )

    def encode(self, sample):
        return self.adapter.encode_sample(sample, device="cpu", dtype=torch.float32)

    def test_four_tasks_use_the_trained_token_layout(self):
        cases = (
            (conditions.force((0.4, 0.5), (0.6, 0.5)), 2),
            (conditions.camera("I"), 80),
            (conditions.hand_pose(torch.full((81, 12, 2), 0.5)), 81 * 12),
            (conditions.robot_pose(torch.full((81, 12, 2), 0.5)), 81 * 12),
        )
        for sample, count in cases:
            with self.subTest(task=sample["task"]):
                tokens, uv, valid, times = self.encode(sample)
                self.assertEqual(tokens.shape, (1, count, 64))
                self.assertEqual(uv.shape, (1, count, 2))
                self.assertEqual(times.shape, (1, count))
                self.assertTrue(valid.all())
                self.assertTrue(torch.isfinite(tokens).all())

    def test_hidden_hand_points_are_removed_from_attention_not_zero_padded(self):
        points = torch.full((81, 12, 2), 0.5)
        mask = torch.ones(81, 12, dtype=torch.bool)
        mask[:, 7:] = False
        points[0, 0] = float("nan")
        tokens, _, valid, _ = self.encode(conditions.hand_pose(points, mask))
        self.assertEqual(tokens.shape[1], 81 * 7 - 1)
        self.assertTrue(valid.all())
        self.assertTrue(torch.isfinite(tokens).all())

    def test_camera_native_midpoints_and_opposites(self):
        from uniaction.camera import opposite_camera_condition

        for key, opposite in zip("WASDIKJL", "SDWAKILJ"):
            sample = conditions.camera(key)
            other = opposite_camera_condition(sample)
            self.assertEqual(other["camera_condition"], opposite)
            torch.testing.assert_close(sample["action_token_time"], torch.arange(80) + 0.5)
            torch.testing.assert_close(other["action_token_time"], sample["action_token_time"])
        self.assertEqual(conditions.camera("I")["action_token_id"][0].item(), 4)
        self.assertEqual(conditions.camera("K")["action_token_id"][0].item(), 5)

    def test_robot_point_order_is_frame_major(self):
        sample = conditions.robot_pose(torch.rand(81, 12, 2))
        torch.testing.assert_close(sample["action_token_id"][:24], torch.arange(12).repeat(2))
        torch.testing.assert_close(
            sample["action_token_time"][:24], torch.tensor([0.0] * 12 + [1.0] * 12)
        )

    def test_rejects_invalid_user_inputs(self):
        for start, end in (
            ((0.0, 0.0), (0.0, 0.0)),
            ((-1.0, 0.0), (0.2, 0.3)),
            ((0.0, 0.0), (float("nan"), 0.2)),
        ):
            with self.assertRaises(ValueError):
                conditions.force(start, end)
        with self.assertRaises(ValueError):
            conditions.robot_pose(torch.full((81, 12, 2), 2.0))
        with self.assertRaises(ValueError):
            conditions.hand_pose(torch.full((81, 12, 2), float("nan")))
        with self.assertRaises(ValueError):
            conditions.camera("Q")

    def test_npz_is_a_non_executable_input_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition.npz"
            np.savez(path, task="camera", key="L")
            self.assertEqual(conditions.load_condition(path)["camera_condition"], "L")
            np.savez(path, task=np.array({"bad": True}, dtype=object))
            with self.assertRaises(ValueError):
                conditions.load_condition(path)

    def test_release_geometry_and_sampling_are_explicit(self):
        config = make_config("/tmp/base", "/tmp/weights")
        for config in (
            replace(config, evaluation=replace(config.evaluation, steps=0)),
            replace(config, model=replace(config.model, num_frames=80)),
        ):
            with self.assertRaises(ValueError):
                config.validate()

    def test_release_requires_weights_and_matching_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = release_metadata()
            (root / "release.json").write_text(json.dumps(metadata))
            self.assertEqual(read_release(root), metadata)
            for stage in ("stage1", "stage2"):
                (root / (stage + ".safetensors")).write_bytes(b"invalid weights")
            verify_checkpoints(root / "stage1.safetensors", root / "stage2.safetensors")
            (root / "stage1.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                verify_checkpoints(root / "stage1.safetensors", root / "stage2.safetensors")
            metadata["action_reads_video"] = False
            (root / "release.json").write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                read_release(root)


if __name__ == "__main__":
    unittest.main()
