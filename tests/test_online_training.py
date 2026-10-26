import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from uniaction.training.online import OnlineEncoder
from uniaction.training.workflow import training_command


class OnlineTrainingTests(unittest.TestCase):
    def test_encoder_keeps_preprocessors_and_matches_cached_tensors(self):
        pipe = SimpleNamespace(device=torch.device("cpu"))
        model = SimpleNamespace(height=480, width=832, num_frames=81)
        context = torch.ones(1, 512, 4096, dtype=torch.bfloat16)
        visual = dict(
            precomputed_input_latents=torch.ones(1, 16, 21, 60, 104, dtype=torch.bfloat16),
            precomputed_y=torch.zeros(1, 20, 21, 60, 104, dtype=torch.bfloat16),
            precomputed_context=context,
        )
        sample = dict(
            video=["frames"],
            prompt="scene",
            task="force",
            action_token_uv=torch.tensor([[0.4, 0.5], [0.6, 0.5]]),
            action_token_time=torch.zeros(2),
            action_token_id=torch.tensor([0, 3]),
            action_normalization="token_list",
        )

        def encode(*args, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(args[2], sample["video"])
            torch.testing.assert_close(kwargs["precomputed_context"], context)
            return visual

        with (
            patch(
                "uniaction.training.online.load_preprocessors", return_value=(pipe, model)
            ) as load,
            patch("uniaction.training.online.encode_prompt_context", return_value=context) as text,
            patch("uniaction.training.online.encode_training_video", side_effect=encode) as video,
        ):
            encoder = OnlineEncoder("/model", "cpu")
            for _ in range(2):
                result = encoder(sample)
                self.assertNotIn("video", result)
                for key, tensor in visual.items():
                    torch.testing.assert_close(result[key], tensor, rtol=0, atol=0)
                torch.testing.assert_close(result["action_token_uv"], sample["action_token_uv"])
                self.assertEqual(result["task"], sample["task"])
                self.assertEqual(result["precomputed_num_frames"], 81)
            self.assertIs(encoder.pipe, pipe)
            load.assert_called_once()
            self.assertEqual(text.call_count, 2)
            self.assertEqual(video.call_count, 2)

    def test_both_stages_forward_raw_sources_or_cache_exclusively(self):
        for raw in (True, False):
            args = SimpleNamespace(
                base_model="base",
                sources="raw.json" if raw else None,
                cache_manifest=None if raw else "cache.json",
                workers=2,
                save_every=250,
                bf16_manifest=None,
                cache_dir=None,
                max_updates=None,
            )
            for stage in (1, 2):
                cmd = training_command(args, Path("/run"), stage, 2, False)
                self.assertEqual("--sources" in cmd, raw)
                self.assertEqual("--cache-manifest" in cmd, not raw)
                self.assertEqual("--stage1-checkpoint" in cmd, stage == 2)

    def test_export_omits_host_paths_and_internal_provenance(self):

        from uniaction.cli.export import export

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in (1, 2):
                run = root / f"stage{stage}"
                run.mkdir()
                (run / f"stage{stage}.safetensors").write_bytes(b"weights")
                (run / "status.json").write_text(json.dumps(dict(status="complete")))
                identity = dict(
                    stage=stage,
                    seed=1,
                    base_model="/private/model",
                    inputs=dict(private_path="/private/research"),
                )
                if stage == 2:
                    identity["stage1_checkpoint"] = str(
                        (root / "stage1/stage1.safetensors").resolve()
                    )
                (run / "identity.json").write_text(json.dumps(identity))
            output = export(root / "stage1", root / "stage2", root / "export")
            result = json.loads((output / "training.json").read_text())
            self.assertEqual(
                result, dict(stage1=dict(stage=1, seed=1), stage2=dict(stage=2, seed=1))
            )


if __name__ == "__main__":
    unittest.main()
