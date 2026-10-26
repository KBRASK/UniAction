import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uniaction.training.data import COUNTS
import torch
from safetensors.torch import load_file

from uniaction.training.prepare import encode_sources, encode_stream


class CacheSourcesTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for name, count in COUNTS.items():
            (root / name).mkdir()
            metadata = root / (name + ".csv")
            metadata.write_text("video,prompt\nexample.mp4,scene\n")
            rows.append(dict(name=name, source_root=name, metadata=metadata.name, samples=count))
        manifest = root / "training-sources.json"
        manifest.write_text(json.dumps(dict(schema="uniaction-training-sources", streams=rows)))
        return SimpleNamespace(
            sources=manifest, base_model=root / "model", output=root / "cache", device="cuda"
        ), rows

    def test_interrupted_cache_reuses_prompts_and_resumes_without_reencoding_samples(self):
        samples = [
            dict(prompt=p, video=i, action=torch.ones(2, 2))
            for i, p in enumerate(("shared", "shared", "different"))
        ]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata = root / "metadata.csv"
            metadata.write_text("video,prompt\n")
            args = SimpleNamespace(
                stream="hand",
                source_root=root,
                metadata=metadata,
                base_model=root / "base",
                output=root / "cache",
                device="cpu",
            )
            fail = True

            def encode(pipe, model, video, prompt, precomputed_context):
                if fail and video == 1:
                    raise RuntimeError("interrupted")
                return dict(precomputed_context=precomputed_context, latents=torch.ones(1) * video)

            with (
                patch("uniaction.training.prepare.source_dataset", return_value=samples),
                patch(
                    "uniaction.training.prepare.load_preprocessors",
                    return_value=(object(), object()),
                ) as load,
                patch(
                    "uniaction.training.prepare.encode_prompt_context",
                    side_effect=lambda _, p: torch.full((1, 2), float(len(p))),
                ) as prompt,
                patch(
                    "uniaction.training.prepare.encode_training_video", side_effect=encode
                ) as video,
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    encode_stream(args)
                fail = False
                encode_stream(args)
                self.assertEqual([c.args[2] for c in video.call_args_list], [0, 1, 1, 2])
                self.assertEqual(
                    [c.args[1] for c in prompt.call_args_list], ["shared", "different", ""]
                )
                index = json.loads((args.output / "index.json").read_text())["records"]
                self.assertEqual(index[0]["context"], index[1]["context"])
                self.assertNotEqual(index[1]["context"], index[2]["context"])
                for row, sample in zip(index, samples):
                    actual = load_file(str(args.output / row["context"]))["context"]
                    self.assertTrue(
                        torch.equal(actual, torch.full((1, 2), float(len(sample["prompt"]))))
                    )
                load.reset_mock()
                video.reset_mock()
                prompt.reset_mock()
                encode_stream(args)
                load.assert_not_called()
                video.assert_not_called()
                prompt.assert_not_called()

    def test_encodes_all_streams_and_builds_resumable_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, rows = self.fixture(root)
            with (
                patch("uniaction.training.prepare.encode_stream") as encode,
                patch("uniaction.training.prepare.cache_manifest") as manifest,
            ):
                encode_sources(args)
            self.assertEqual(
                [c.args[0].stream for c in encode.call_args_list], [r["name"] for r in rows]
            )
            for call, row in zip(encode.call_args_list, rows):
                self.assertEqual(call.args[0].source_root, root / row["source_root"])
                self.assertEqual(call.args[0].output, args.output / row["name"])
            manifest.assert_called_once_with(
                args.output, args.output / "training.json", resume=True
            )

    def test_rejects_missing_last_metadata_before_any_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, rows = self.fixture(root)
            (root / rows[-1]["metadata"]).unlink()
            with patch("uniaction.training.prepare.encode_stream") as encode:
                with self.assertRaises(FileNotFoundError):
                    encode_sources(args)
                encode.assert_not_called()

    def test_rejects_duplicate_streams_and_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            args, rows = self.fixture(Path(d))
            for invalid in (
                [rows[0]] * len(rows),
                [dict(rows[0], metadata="../outside.csv"), *rows[1:]],
            ):
                args.sources.write_text(
                    json.dumps(dict(schema="uniaction-training-sources", streams=invalid))
                )
                with patch("uniaction.training.prepare.encode_stream") as encode:
                    with self.assertRaises(ValueError):
                        encode_sources(args)
                    encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
