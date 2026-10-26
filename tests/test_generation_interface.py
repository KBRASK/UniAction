import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from uniaction import UniActionPipeline, conditions
from uniaction.config import make_config


class GenerationInterfaceTests(unittest.TestCase):
    def test_shared_guidance_options_cover_every_action_type(self):
        image = Image.new("RGB", (832, 480))
        frames = [image] * 81
        uv = np.full((81, 12, 2), 0.5, np.float32)
        actions = [
            conditions.force((0.4, 0.5), (0.6, 0.5)),
            conditions.camera("W"),
            conditions.hand_pose(uv),
            conditions.robot_pose(uv),
        ]
        pipe = UniActionPipeline(make_config("/base", "/weights"), None, None)
        with (
            patch("uniaction.pipeline.generate_sample", return_value=frames) as generate,
            patch("uniaction.pipeline.generate_camera_sample", return_value=frames) as camera,
            patch(
                "uniaction.guidance.generate_counterfactual", return_value=frames
            ) as counterfactual,
        ):
            for mode, text, action in [
                ("separate", 1.0, 2.0),
                ("joint", 3.0, 3.0),
                ("counterfactual", 3.0, 3.0),
            ]:
                for condition in actions:
                    with self.subTest(mode=mode, task=condition["semantic_task"]):
                        output = pipe.generate(
                            image,
                            "A static scene.",
                            condition,
                            guidance=mode,
                            text_guidance=text,
                            action_guidance=action,
                        )
                        self.assertEqual(len(output), 81)
                        call = (
                            counterfactual
                            if mode == "counterfactual"
                            else camera
                            if condition["semantic_task"] == "camera"
                            else generate
                        )
                        self.assertEqual(call.call_args.kwargs["text_cfg_scale"], text)
                        self.assertEqual(call.call_args.kwargs["action_cfg_scale"], action)
                        if mode != "counterfactual":
                            self.assertEqual(
                                call.call_args.kwargs["cfg_mode"],
                                "camera_joint" if call is camera and mode == "joint" else mode,
                            )

    def test_incompatible_shared_scales_fail_before_model_execution(self):
        pipe = UniActionPipeline(make_config("/base", "/weights"), None, None)
        image = Image.new("RGB", (832, 480))
        with patch("uniaction.pipeline.generate_camera_sample") as generate:
            with self.assertRaisesRegex(ValueError, "equal"):
                pipe.generate(
                    image,
                    "A room.",
                    conditions.camera("W"),
                    guidance="joint",
                    text_guidance=1.0,
                    action_guidance=2.0,
                )
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
