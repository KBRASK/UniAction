import importlib.util
import unittest

if any(importlib.util.find_spec(name) is None for name in ("accelerate",)):
    raise unittest.SkipTest("Optional dependencies are not installed")

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from accelerate import Accelerator

from uniaction import conditions
from uniaction.backbones.wan.dit import WanModel
from uniaction.backbones.wan.pipeline import WanPipeline
from uniaction.checkpoints import read_release, verify_checkpoints
from uniaction.cli.export import export
from uniaction.config import ModelConfig
from uniaction.training.config import TrainingConfig
from uniaction.training.model import UniActionTrainingModule
from uniaction.training.recovery import (
    atomic_json,
    export_trainable_model_state,
    restore_rng_state,
    restore_trainable_model_state,
)
from uniaction.training.runner import save_checkpoint, load_checkpoint


class TinyModelConfig(ModelConfig):
    @property
    def profile(self):
        return SimpleNamespace(num_blocks=2, expert_names=("dit",))


def tiny_pipe(*args, **kwargs):
    pipe = WanPipeline(torch_dtype=torch.bfloat16, device="cpu")
    pipe.dit = WanModel(
        dim=128,
        in_dim=36,
        ffn_dim=128,
        out_dim=16,
        text_dim=4096,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=1,
        num_layers=2,
        has_image_input=False,
        require_clip_embedding=False,
    ).to(torch.bfloat16)
    from uniaction.model.wan import attach_wan_attention_runtime

    attach_wan_attention_runtime(pipe.dit, kwargs["attention_runtime"])
    return pipe


def tiny_module(stage, stage1=None):
    config = TrainingConfig(
        model=TinyModelConfig(Path("."), Path("."), Path(".")),
        seed=1 if stage == 1 else 101,
        stage1=SimpleNamespace(prompt_dropout=0),
        stage2=SimpleNamespace(stage1_checkpoint=stage1, prompt_dropout=0),
        artifacts=SimpleNamespace(),
    )
    with patch("uniaction.training.model.load_wan_pipeline", tiny_pipe):
        module = UniActionTrainingModule(
            config, stage, attention_backend="legacy", attention_fallback=False
        )
    module.resolve_attention_backend("cpu")
    return module


def sample():
    result = conditions.force((0.45, 0.5), (0.55, 0.5))
    result.update(
        precomputed_input_latents=torch.randn(1, 16, 3, 4, 4).bfloat16(),
        precomputed_y=torch.randn(1, 20, 3, 4, 4).bfloat16(),
        precomputed_context=torch.randn(1, 4, 4096).bfloat16(),
    )
    return result


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_stages_backward_and_restore(self):
        from uniaction.training.recovery import atomic_safetensors

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage1.safetensors"
            first = tiny_module(1)
            loss = first(sample())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None for p in first.trainable_parameters()))
            atomic_safetensors(checkpoint, export_trainable_model_state(first, first.state_dict()))
            second = tiny_module(2, checkpoint)
            loss = second(sample())
            loss.backward()
            second.clear_staged_action()
            self.assertTrue(torch.isfinite(loss))
            trainable = dict((n, p) for n, p in second.named_parameters() if p.requires_grad)
            self.assertTrue(
                any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable.values())
            )
            saved = {
                k: v.clone()
                for k, v in export_trainable_model_state(second, second.state_dict()).items()
            }
            self.assertTrue(any(k.endswith("selfattn_action_log_reference_count") for k in saved))
            with torch.no_grad():
                next(second.trainable_parameters()).add_(1)
            restore_trainable_model_state(second, saved)
            for k, v in export_trainable_model_state(second, second.state_dict()).items():
                torch.testing.assert_close(v, saved[k], rtol=0, atol=0)

    def test_recovery_restores_exact_next_update(self):
        accelerator = Accelerator()
        with tempfile.TemporaryDirectory() as directory:
            module = tiny_module(1)
            optimizer = torch.optim.AdamW(module.trainable_parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: (u + 1) / 100)
            inputs = sample()

            def step():
                optimizer.zero_grad()
                loss = module(inputs)
                loss.backward()
                optimizer.step()
                scheduler.step()
                module.clear_staged_action()
                return loss.detach().clone()

            step()
            save_checkpoint(
                accelerator, module, optimizer, scheduler, Path(directory), 1, 1, {"test": True}
            )
            expected_loss = step()
            expected_weights = {
                k: v.clone()
                for k, v in export_trainable_model_state(module, module.state_dict()).items()
            }
            update, rank = load_checkpoint(
                accelerator, module, optimizer, scheduler, Path(directory), 1, {"test": True}
            )
            self.assertEqual(update, 1)
            restore_rng_state(module, rank)
            torch.testing.assert_close(step(), expected_loss, rtol=0, atol=0)
            for k, v in export_trainable_model_state(module, module.state_dict()).items():
                torch.testing.assert_close(v, expected_weights[k], rtol=0, atol=0)

    def test_resume_preserves_shuffled_data_order(self):
        accelerator = Accelerator()

        def loader():
            return accelerator.prepare(
                torch.utils.data.DataLoader(
                    torch.arange(64),
                    batch_size=1,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(101),
                    num_workers=0,
                )
            )

        expected = [int(x) for x in loader()]
        resumed = [int(x) for x in accelerator.skip_first_batches(loader(), 16)]
        self.assertEqual(expected[16:], resumed)

    def test_exported_weights_require_explicit_opt_in(self):
        from uniaction.config import make_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in (1, 2):
                run = root / str(stage)
                run.mkdir()
                (run / f"stage{stage}.safetensors").write_bytes(bytes([stage]) * 32)
                atomic_json(run / "status.json", {"status": "complete"})
                atomic_json(
                    run / "identity.json",
                    {
                        "stage": stage,
                        "stage1_checkpoint": str((root / "1/stage1.safetensors").resolve()),
                    },
                )
            output = export(root / "1", root / "2", root / "package")
            verify_checkpoints(
                output / "stage1.safetensors", output / "stage2.safetensors", trained=True
            )
            config = make_config(root, output, trained=True)
            config.validate()
            from uniaction.inference import build_pipeline

            class ReachedBackboneLoad(Exception):
                pass

            with patch("uniaction.inference.load_wan_pipeline", side_effect=ReachedBackboneLoad):
                with self.assertRaises(ReachedBackboneLoad):
                    build_pipeline(config, device="cpu", allow_compatibility_runtime=True)
            with self.assertRaises(ValueError):
                read_release(output)
            (output / "stage2.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                verify_checkpoints(
                    output / "stage1.safetensors", output / "stage2.safetensors", trained=True
                )


if __name__ == "__main__":
    unittest.main()
