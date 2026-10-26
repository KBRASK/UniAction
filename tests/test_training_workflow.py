import importlib.util
import unittest

if any(importlib.util.find_spec(name) is None for name in ("accelerate",)):
    raise unittest.SkipTest("Optional dependencies are not installed")

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uniaction.training.recovery import atomic_json, inventory_file
from uniaction.training.runner import finish_run
from uniaction.training.workflow import train_all


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache.json"
        self.cache.write_text("{}")
        self.args = SimpleNamespace(
            base_model=str(self.root / "base"),
            cache_manifest=str(self.cache),
            output=str(self.root / "run"),
            resume=False,
            gpus=2,
            workers=0,
            save_every=1,
            max_updates=2,
            bf16_manifest=None,
            cache_dir=None,
        )
        self.calls = []
        self.pause = None
        self.fail = None
        self.signal_at = None

    def worker(self, command, **kwargs):
        stage = int(command[command.index("--stage") + 1])
        self.calls.append((stage, "--resume" in command))
        if self.fail == stage:
            raise subprocess.CalledProcessError(1, command)
        run = Path(command[command.index("--output") + 1])
        run.mkdir(exist_ok="--resume" in command)
        update = 1 if self.pause == stage else 2
        recovery = run / "recovery" / f"update-{update:08d}"
        recovery.mkdir(parents=True)
        model = recovery / "model.safetensors"
        model.write_bytes(bytes([stage]) * 32)
        identity = dict(stage=stage, optimizer_updates=2, stage1_checkpoint=None)
        if stage == 2:
            checkpoint = Path(command[command.index("--stage1-checkpoint") + 1])
            self.assertEqual(checkpoint, Path(self.args.output) / "stage1/stage1.safetensors")
            identity["stage1_checkpoint"] = str(checkpoint.resolve())
        atomic_json(run / "identity.json", identity)
        inventory = atomic_json(
            recovery / "manifest.json",
            dict(
                update=update,
                identity=identity,
                files={"model.safetensors": inventory_file(model)},
            ),
        )
        atomic_json(run / "recovery/latest.json", dict(directory=recovery.name, manifest=inventory))
        if self.signal_at == stage:
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertTrue(Path(command[command.index("--stop-file") + 1]).exists())
        finish_run(run, stage, recovery, update, 2, self.pause == stage or self.signal_at == stage)
        return subprocess.CompletedProcess(command, 0)

    def execute(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("uniaction.training.workflow.subprocess.run", side_effect=self.worker),
        ):
            train_all(self.args)

    def test_complete_handoff_export_and_noop_resume(self):
        self.execute()
        self.assertEqual(self.calls, [(1, False), (2, False)])
        self.assertTrue((Path(self.args.output) / "checkpoint/release.json").is_file())
        self.args.resume = True
        self.execute()
        self.assertEqual(len(self.calls), 2)

    def test_paused_first_stage_stops_then_resumes_and_hands_off(self):
        self.pause = 1
        self.execute()
        self.assertEqual(self.calls, [(1, False)])
        self.assertFalse((Path(self.args.output) / "stage2").exists())
        self.pause = None
        self.args.resume = True
        self.execute()
        self.assertEqual(self.calls, [(1, False), (1, True), (2, False)])

    def test_resume_second_stage_does_not_repeat_first(self):
        self.pause = 2
        self.execute()
        self.pause = None
        self.args.resume = True
        self.execute()
        self.assertEqual(self.calls, [(1, False), (2, False), (2, True)])

    def test_child_failure_never_launches_second_stage(self):
        self.fail = 1
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        self.assertEqual(self.calls, [(1, False)])

    def test_stop_during_final_update_does_not_start_next_stage(self):
        self.signal_at = 1
        self.execute()
        self.assertEqual(self.calls, [(1, False)])
        self.args.resume = True
        self.signal_at = None
        self.execute()
        self.assertEqual(self.calls, [(1, False), (2, False)])

    def test_changed_input_path_rejects_resume(self):
        self.pause = 1
        self.execute()
        self.args.resume = True
        other_cache = self.cache.with_name("other-cache.json")
        other_cache.write_text(self.cache.read_text())
        self.args.cache_manifest = str(other_cache)
        with self.assertRaisesRegex(ValueError, "same data"):
            self.execute()
        self.assertEqual(len(self.calls), 1)

    def test_corrupt_completed_weights_are_rejected(self):
        self.signal_at = 1
        self.execute()
        (Path(self.args.output) / "stage1/stage1.safetensors").write_bytes(b"bad")
        self.args.resume = True
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(len(self.calls), 1)

    def test_recovery_finishes_export_after_interrupted_finalization(self):
        self.pause = 1
        self.execute()
        run = Path(self.args.output) / "stage1"
        recovery = run / "recovery/update-00000001"
        finish_run(run, 1, recovery, 2, 2, False)
        self.assertTrue((run / "stage1.safetensors").is_file())
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "complete")
        finish_run(run, 1, recovery, 2, 2, False)


if __name__ == "__main__":
    unittest.main()
