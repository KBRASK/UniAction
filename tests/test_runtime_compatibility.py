import subprocess
import sys
import unittest
import warnings
from unittest.mock import patch

import torch

from uniaction.model.attention import AttentionBackendRuntime
from uniaction.runtime_profiles import resolve_runtime_profile, validate_runtime_summary


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_fa4_hardware_selection(self):
        for name, capability, supported in (
            ("NVIDIA H100 80GB HBM3", (9, 0), True),
            ("NVIDIA H200", (9, 0), True),
            ("NVIDIA RTX PRO 6000 Blackwell", (12, 0), True),
            ("NVIDIA A100", (8, 0), False),
            ("NVIDIA RTX 5090", (12, 0), False),
        ):
            with (
                self.subTest(name=name),
                patch("torch.cuda.get_device_name", return_value=name),
                patch("torch.cuda.get_device_capability", return_value=capability),
            ):
                if supported:
                    AttentionBackendRuntime._check_hardware("fa4", torch.device("cuda"))
                else:
                    with self.assertRaisesRegex(RuntimeError, "FA4 is not supported"):
                        AttentionBackendRuntime._check_hardware("fa4", torch.device("cuda"))

    def test_missing_flash_extensions_report_sdpa_fallback_once(self):
        runtime = AttentionBackendRuntime(requested="fa4")
        with patch.object(
            runtime, "_probe", side_effect=ImportError("extension unavailable")
        ) as probe:
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                self.assertEqual(runtime.resolve("cpu"), "legacy")
                self.assertEqual(runtime.resolve("cpu"), "legacy")
            self.assertEqual(probe.call_count, 3)
            self.assertEqual(sum("fallback selected: SDPA" in str(m.message) for m in messages), 1)
            self.assertTrue(any("FA4 unavailable" in str(m.message) for m in messages))
        value = torch.randn(1, 5, 32)
        result = runtime.plain_attention(value, value, value, num_heads=4)
        self.assertTrue(torch.isfinite(result).all())

    def test_generic_gpu_profile_accepts_selected_fallback(self):
        for stage in (1, 2):
            profile = resolve_runtime_profile(
                stage=stage,
                device_name="NVIDIA A100",
                compute_capability=(8, 0),
                model_family="Wan2.2-I2V-A14B-HighNoise",
                num_frames=81,
            )
            self.assertTrue(profile.attention_fallback)
            self.assertEqual(profile.rope_backend, "baseline")
            self.assertEqual(profile.rmsnorm_backend, "baseline")
            runtime = AttentionBackendRuntime(
                requested="fa4",
                selected="legacy",
                require_key_bias=stage == 2,
                fa4_bias_gradient=profile.fa4_bias_gradient,
                fa4_key_bias_layout=profile.fa4_key_bias_layout,
                fa4_split_lse_mix_backend=profile.fa4_split_lse_mix_backend,
            )
            validate_runtime_summary(
                profile,
                attention=runtime.summary(),
                primitive_backends={
                    "dit": {"rope_backend": "baseline", "rmsnorm_backend": "baseline"}
                },
                regional_compile=None,
                require_call_coverage=False,
            )

    def test_evaluation_help_without_optional_dependencies(self):
        program = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'skimage', 'cv2', 'scipy', 'lpips', 'cotracker'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockOptional())
from uniaction.cli.evaluate import main
sys.argv = ['uniaction-eval', *sys.argv[1:]]
main()
"""
        for args in (("--help",), ("similarity", "--help")):
            result = subprocess.run(
                [sys.executable, "-c", program, *args], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
