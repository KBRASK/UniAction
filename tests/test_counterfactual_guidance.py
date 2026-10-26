import unittest

import numpy as np
import torch

from uniaction import conditions
from uniaction.guidance import counterfactual


class CounterfactualTests(unittest.TestCase):
    def test_force_reverses_direction_without_moving_contact_or_mutating_input(self):
        original = conditions.force((0.5, 0.5), (0.6, 0.4))
        before = original["action_token_uv"].clone()
        reference = counterfactual(original)
        torch.testing.assert_close(reference["action_token_uv"][0], before[0])
        torch.testing.assert_close(
            reference["action_token_uv"][1] - before[0], before[0] - before[1]
        )
        torch.testing.assert_close(original["action_token_uv"], before)

    def test_hand_reference_retains_initial_visibility(self):
        uv = np.full((81, 12, 2), 0.5, np.float32)
        uv[:, :, 0] += np.arange(81)[:, None] / 1000
        visible = np.ones((81, 12), bool)
        visible[0, 2] = False
        original = conditions.hand_pose(uv, visible)
        reference = counterfactual(original)
        self.assertTrue(torch.equal(reference["action"][0], reference["action"][80]))
        self.assertFalse(reference["action_point_time_mask"][:, 2].any())
        self.assertTrue(original["action_point_time_mask"][1:, 2].all())

    def test_robot_holds_first_observation_without_inventing_timestamps(self):
        uv = np.full((81, 12, 2), 0.5, np.float32)
        uv[:, :, 0] += np.arange(81)[:, None] / 1000
        visible = np.zeros((81, 12), bool)
        visible[[2, 15, 70], 8] = True
        original = conditions.robot_pose(uv, visible)
        reference = counterfactual(original)
        torch.testing.assert_close(reference["action_token_time"], original["action_token_time"])
        torch.testing.assert_close(reference["action_token_id"], original["action_token_id"])
        torch.testing.assert_close(
            reference["action_token_uv"], original["action_token_uv"][0:1].expand(3, 2)
        )

    def test_camera_reverses_each_key(self):
        for key, opposite in zip("WASDIKJL", "SDWAKILJ"):
            with self.subTest(key=key):
                reference = counterfactual(conditions.camera(key))
                self.assertEqual(reference["camera_condition"], opposite)

    def test_outside_reference_is_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            counterfactual(conditions.force((0.1, 0.5), (0.9, 0.5)))


if __name__ == "__main__":
    unittest.main()
