from __future__ import annotations

import multiprocessing
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from uniaction.model.weight_cache import (
    WeightCacheRequest,
    cache_file_groups,
    cache_files,
)


def _concurrent_stage_worker(
    source: str,
    stage_root: str,
    barrier,
    results,
) -> None:
    try:
        barrier.wait()
        paths, report = cache_files(
            (Path(source),),
            stage_root=Path(stage_root),
            namespace="models/test/dit",
        )
        results.put(
            {
                "cache_hit": report["cache_hit"],
                "target": str(paths[0]),
                "payload": paths[0].read_bytes(),
            }
        )
    except BaseException as error:
        results.put({"error": repr(error)})


class WeightCacheTests(unittest.TestCase):
    def test_insufficient_space_fails_before_reading_or_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"model-payload")
            with patch(
                "uniaction.model.weight_cache.shutil.disk_usage",
                return_value=types.SimpleNamespace(free=1),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "insufficient weight cache capacity before copy",
                ):
                    cache_files(
                        (source,),
                        stage_root=root / "stage",
                        namespace="models/test/dit",
                    )
            objects = root / "stage/objects/models/test/dit"
            self.assertEqual(
                [path for path in objects.iterdir() if not path.name.startswith(".")],
                [],
            )

    def test_first_copy_and_stat_guarded_reuse_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"weight-cache" * 1024)
            first_paths, first = cache_files(
                (source,),
                stage_root=root / "stage",
                namespace="wan/dit",
            )
            self.assertFalse(first["cache_hit"])
            self.assertEqual(first_paths[0].read_bytes(), source.read_bytes())
            second_paths, second = cache_files(
                (source,),
                stage_root=root / "stage",
                namespace="wan/dit",
            )
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second_paths, first_paths)

    def test_changed_published_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"original")
            staged_paths, _report = cache_files(
                (source,),
                stage_root=root / "stage",
                namespace="wan/dit",
            )
            staged_paths[0].write_bytes(b"tampered-longer")
            with self.assertRaisesRegex(ValueError, "changed after publication"):
                cache_files(
                    (source,),
                    stage_root=root / "stage",
                    namespace="wan/dit",
                )

    def test_source_stat_change_uses_a_new_cache_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"first")
            first_paths, first = cache_files(
                (source,),
                stage_root=root / "stage",
                namespace="preprocessors/t5",
            )
            source.write_bytes(b"second-version")
            os.utime(source, None)
            second_paths, second = cache_files(
                (source,),
                stage_root=root / "stage",
                namespace="preprocessors/t5",
            )
            self.assertNotEqual(first["cache_id"], second["cache_id"])
            self.assertNotEqual(first_paths, second_paths)
            self.assertEqual(second_paths[0].read_bytes(), b"second-version")

    def test_rejects_ambiguous_names_and_unsafe_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a" / "same.bin"
            second = root / "b" / "same.bin"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            with self.assertRaisesRegex(ValueError, "unique source basenames"):
                cache_files(
                    (first, second),
                    stage_root=root / "stage",
                    namespace="wan/dit",
                )
            with self.assertRaisesRegex(ValueError, "invalid.*namespace"):
                cache_files(
                    (first,),
                    stage_root=root / "stage",
                    namespace="../escape",
                )

    def test_rejects_symlinked_or_group_readable_stage_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"payload")
            actual = root / "actual"
            actual.mkdir(mode=0o700)
            linked = root / "linked"
            linked.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                cache_files(
                    (source,),
                    stage_root=linked,
                    namespace="models/test/dit",
                )

            exposed = root / "exposed"
            exposed.mkdir(mode=0o755)
            exposed.chmod(0o755)
            with self.assertRaisesRegex(PermissionError, "mode 0700"):
                cache_files(
                    (source,),
                    stage_root=exposed,
                    namespace="models/test/dit",
                )

    def test_concurrent_processes_publish_exactly_one_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            payload = b"concurrent-weight-cache" * (256 * 1024)
            source.write_bytes(payload)
            context = multiprocessing.get_context("fork")
            barrier = context.Barrier(4)
            results = context.Queue()
            processes = [
                context.Process(
                    target=_concurrent_stage_worker,
                    args=(str(source), str(root / "stage"), barrier, results),
                )
                for _ in range(4)
            ]
            for process in processes:
                process.start()
            reports = [results.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
            self.assertFalse(any("error" in report for report in reports), reports)
            self.assertEqual(
                sum(not report["cache_hit"] for report in reports),
                1,
            )
            self.assertEqual(len({report["target"] for report in reports}), 1)
            self.assertTrue(all(report["payload"] == payload for report in reports))

    def test_stale_unpublished_directory_is_removed_under_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"payload")
            paths, report = cache_files(
                (source,),
                stage_root=root / "stage",
                namespace="models/test/dit",
            )
            target = paths[0].parent
            target.rename(root / "published-away")
            stale = target.parent / ".pending-interrupted"
            stale.mkdir()
            (stale / "partial.bin").write_bytes(b"partial")
            _paths, rebuilt = cache_files(
                (source,),
                stage_root=root / "stage",
                namespace="models/test/dit",
            )
            self.assertFalse(rebuilt["cache_hit"])
            self.assertEqual(rebuilt["stale_temporary_directories_removed"], 1)
            self.assertFalse(stale.exists())

    def test_independent_file_groups_are_dispatched_concurrently(self) -> None:
        barrier = threading.Barrier(3)

        def fake_stage(source_files, *, stage_root, namespace):
            del stage_root
            barrier.wait(timeout=5)
            paths = tuple(Path(path) for path in source_files)
            return paths, {"namespace": namespace}

        requests = {
            name: WeightCacheRequest(
                source_files=(Path(f"{name}.bin"),),
                namespace=f"models/test/{name}",
            )
            for name in ("dit", "text_encoder", "vae")
        }
        with patch(
            "uniaction.model.weight_cache.cache_files",
            side_effect=fake_stage,
        ):
            staged = cache_file_groups(
                requests,
                stage_root=Path("unused"),
            )
        self.assertEqual(tuple(staged), tuple(requests))
        self.assertEqual(
            {value[1]["namespace"] for value in staged.values()},
            {request.namespace for request in requests.values()},
        )


if __name__ == "__main__":
    unittest.main()
