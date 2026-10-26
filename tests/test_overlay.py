import importlib.util
import unittest

if any(importlib.util.find_spec(name) is None for name in ("cv2",)):
    raise unittest.SkipTest("Optional dependencies are not installed")

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from uniaction import UniActionPipeline, conditions
from uniaction.config import make_config
from uniaction.overlay import overlay_frames


class OverlayTests(unittest.TestCase):
    def test_robot_links_visible_skeleton_points(self):
        points = np.full((81, 12, 2), 0.2, np.float32)
        points[:, 7] = (0.6, 0.3)
        points[:, 8] = (0.7, 0.3)
        visible = np.zeros((81, 12), bool)
        visible[:, 7:9] = True
        raw = Image.new("RGB", (832, 480), (100, 100, 100))
        rendered = overlay_frames([raw] * 81, conditions.robot_pose(points, visible))
        self.assertNotEqual(rendered[0].getpixel((541, 144)), (100, 100, 100))
        self.assertEqual(rendered[0].getpixel((166, 96)), (100, 100, 100))

    def test_hidden_hand_points_remain_hidden_and_visible_points_are_transparent(self):
        points = np.full((81, 12, 2), 0.5, np.float32)
        points[:, 0] = (0.25, 0.25)
        visible = np.zeros((81, 12), bool)
        visible[:, 0] = True
        raw = Image.new("RGB", (832, 480), (100, 100, 100))
        rendered = overlay_frames([raw] * 81, conditions.hand_pose(points, visible))
        self.assertEqual(rendered[0].getpixel((208, 120)), (30, 30, 208))
        self.assertEqual(rendered[0].getpixel((416, 240)), (100, 100, 100))
        self.assertEqual(raw.getpixel((208, 120)), (100, 100, 100))

    def test_camera_uses_command_direction_and_final_interval_is_inactive(self):
        raw = Image.new("RGB", (832, 480), (80, 80, 80))
        w = overlay_frames([raw] * 81, conditions.camera("W"))
        s = overlay_frames([raw] * 81, conditions.camera("S"))
        self.assertFalse(np.array_equal(w[0], s[0]))
        np.testing.assert_array_equal(w[80], s[80])
        self.assertFalse(np.array_equal(w[0], w[80]))
        self.assertEqual(w[0].getpixel((416, 240)), raw.getpixel((416, 240)))

    def test_pipeline_saves_two_playable_outputs_from_one_model_call(self):
        raw = Image.new("RGB", (832, 480), (80, 80, 80))
        frames = [raw] * 81
        pipe = UniActionPipeline(make_config("/base", "/weights"), None, None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.mp4"
            with patch("uniaction.pipeline.generate_sample", return_value=frames) as generate:
                result = pipe.generate(
                    raw,
                    "A tabletop.",
                    conditions.force((0.4, 0.5), (0.6, 0.5)),
                    output=output,
                    overlay=True,
                )
            self.assertIs(result, frames)
            generate.assert_called_once()
            decoded = []
            for path in [output, output.with_name("result-overlay.mp4")]:
                with imageio.get_reader(str(path), input_params=["-threads", "1"]) as reader:
                    self.assertEqual(reader.count_frames(), 81)
                    self.assertEqual(reader.get_meta_data()["fps"], 16)
                    decoded.append(reader.get_data(0))
            self.assertFalse(np.array_equal(*decoded))
            before = output.read_bytes()
            with (
                patch("uniaction.pipeline.generate_sample") as generate,
                self.assertRaisesRegex(ValueError, "already exists"),
            ):
                pipe.generate(
                    raw,
                    "A tabletop.",
                    conditions.force((0.4, 0.5), (0.6, 0.5)),
                    output=output,
                    overlay=True,
                )
            generate.assert_not_called()
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
