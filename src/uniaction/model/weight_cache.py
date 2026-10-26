from __future__ import annotations

import errno
import fcntl
import uuid
import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


STAGE_FORMAT = "uniaction-weight-cache-v1"
STAGE_FREE_SPACE_RESERVE_BYTES = 256 * 1024 * 1024
_SAFE_NAMESPACE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_STAT_KEYS = ("device", "inode", "size", "mtime_ns", "ctime_ns")


@dataclass(frozen=True)
class WeightCacheRequest:
    source_files: tuple[Path, ...]
    namespace: str


def _stat_receipt(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _validate_namespace(namespace: str) -> tuple[str, ...]:
    parts = PurePosixPath(namespace).parts
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or any(_SAFE_NAMESPACE_PART.fullmatch(part) is None for part in parts)
    ):
        raise ValueError(f"invalid weight cache namespace: {namespace!r}")
    return parts


def _source_inventory(
    source_files: Iterable[Path],
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    sources = tuple(Path(path).expanduser().resolve(strict=True) for path in source_files)
    if not sources:
        raise ValueError("weight cache file list is empty")
    names = [path.name for path in sources]
    if len(names) != len(set(names)):
        raise ValueError("weight cache requires unique source basenames")
    inventory = []
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory.append(
            {
                "name": path.name,
                "path": str(path),
                "stat": _stat_receipt(path),
            }
        )
    return sources, inventory


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid weight cache receipt: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"weight cache receipt must be an object: {path}")
    return payload


def _validate_published_stage(
    target: Path,
    *,
    cache_id: str,
    namespace: str,
    source_inventory: list[dict[str, Any]],
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    receipt_path = target / "receipt.json"
    receipt = _load_receipt(receipt_path)
    if (
        receipt.get("format") != STAGE_FORMAT
        or receipt.get("cache_id") != cache_id
        or receipt.get("namespace") != namespace
        or receipt.get("source_files") != source_inventory
    ):
        raise ValueError(f"weight cache receipt drift: {receipt_path}")
    staged_entries = receipt.get("staged_files")
    if not isinstance(staged_entries, list) or len(staged_entries) != len(source_inventory):
        raise ValueError(f"invalid staged-file inventory: {receipt_path}")
    staged_paths = []
    for source, staged in zip(source_inventory, staged_entries, strict=True):
        if not isinstance(staged, dict) or staged.get("name") != source["name"]:
            raise ValueError(f"staged-file ordering mismatch: {receipt_path}")
        path = target / str(staged["name"])
        if not path.is_file():
            raise FileNotFoundError(f"staged model file is missing: {path}")
        actual_stat = _stat_receipt(path)
        expected_stat = staged.get("stat")
        if not isinstance(expected_stat, dict) or any(
            int(expected_stat.get(key, -1)) != actual_stat[key] for key in _STAT_KEYS
        ):
            raise ValueError(f"staged model file changed after publication: {path}")
        staged_paths.append(path)
    return tuple(staged_paths), receipt


def _copy_file(source: Path, destination: Path) -> None:
    source_size = source.stat().st_size
    available_bytes = shutil.disk_usage(destination.parent).free
    required_bytes = source_size + STAGE_FREE_SPACE_RESERVE_BYTES
    if available_bytes < required_bytes:
        raise OSError(
            errno.ENOSPC,
            "insufficient weight cache capacity before copy: "
            f"required_bytes={required_bytes} source_bytes={source_size} "
            f"reserve_bytes={STAGE_FREE_SPACE_RESERVE_BYTES} "
            f"available_bytes={available_bytes} destination={destination.parent}",
        )
    with source.open("rb") as reader, destination.open("xb") as writer:
        if hasattr(os, "posix_fallocate"):
            try:
                os.posix_fallocate(writer.fileno(), 0, source_size)
            except OSError as error:
                if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
                    raise OSError(
                        error.errno,
                        "weight cache reservation failed before copy: "
                        f"source_bytes={source_size} destination={destination}",
                    ) from error
        while chunk := reader.read(16 * 1024 * 1024):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_stale_temporary_directories(
    object_parent: Path,
) -> int:

    removed = 0
    prefix = ".pending-"
    for candidate in object_parent.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"unsafe stale weight cache entry: {candidate}")
        shutil.rmtree(candidate)
        removed += 1
    if removed:
        _fsync_directory(object_parent)
    return removed


def cache_files(
    source_files: Iterable[Path],
    *,
    stage_root: Path,
    namespace: str,
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    started = time.perf_counter()
    namespace_parts = _validate_namespace(namespace)
    sources, source_inventory = _source_inventory(source_files)
    identity = dict(format=STAGE_FORMAT, namespace=namespace, source_files=source_inventory)
    requested = Path(stage_root).expanduser()
    if requested.is_symlink():
        raise ValueError(f"weight cache root must not be a symlink: {requested}")
    requested.mkdir(parents=True, exist_ok=True, mode=0o700)
    stat = requested.stat()
    if stat.st_uid != os.geteuid():
        raise PermissionError(f"weight cache root is owned by uid {stat.st_uid}")
    if stat.st_mode & 0o077:
        raise PermissionError(f"weight cache root must have mode 0700: {requested}")
    object_parent = requested.resolve().joinpath("objects", *namespace_parts)
    object_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_started = time.perf_counter()
    with (object_parent / ".stage.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lock_acquired = time.perf_counter()
        removed = _remove_stale_temporary_directories(object_parent)
        target = None
        for candidate in sorted(object_parent.iterdir()):
            if candidate.name.startswith("."):
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise ValueError(f"invalid weight cache entry: {candidate}")
            receipt = _load_receipt(candidate / "receipt.json")
            if all(receipt.get(key) == value for key, value in identity.items()):
                target = candidate
                break
        cache_hit = target is not None
        copy_seconds = 0.0
        if cache_hit:
            cache_id = target.name
        else:
            cache_id = uuid.uuid4().hex
            target = object_parent / cache_id
            temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=object_parent))
            copy_started = time.perf_counter()
            try:
                staged_entries = []
                for source, entry in zip(sources, source_inventory, strict=True):
                    destination = temporary / source.name
                    _copy_file(source, destination)
                    if _stat_receipt(source) != entry["stat"]:
                        raise ValueError(f"model source changed while staging: {source}")
                    if destination.stat().st_size != entry["stat"]["size"]:
                        raise ValueError(f"incomplete staged model file: {destination}")
                    staged_entries.append(dict(name=source.name, stat=_stat_receipt(destination)))
                receipt = dict(
                    identity,
                    cache_id=cache_id,
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    staged_files=staged_entries,
                )
                with (temporary / "receipt.json").open("x", encoding="utf-8") as handle:
                    json.dump(receipt, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_directory(temporary)
                os.replace(temporary, target)
                _fsync_directory(object_parent)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            copy_seconds = time.perf_counter() - copy_started
        staged_paths, receipt = _validate_published_stage(
            target, cache_id=cache_id, namespace=namespace, source_inventory=source_inventory
        )
    return staged_paths, dict(
        format=STAGE_FORMAT,
        cache_id=cache_id,
        namespace=namespace,
        target=str(target),
        cache_hit=cache_hit,
        file_count=len(staged_paths),
        staged_bytes=sum(p.stat().st_size for p in staged_paths),
        lock_wait_seconds=lock_acquired - lock_started,
        copy_seconds=copy_seconds,
        total_seconds=time.perf_counter() - started,
        stale_temporary_directories_removed=removed,
    )


def cache_file_groups(
    requests: Mapping[str, WeightCacheRequest],
    *,
    stage_root: Path,
) -> dict[str, tuple[tuple[Path, ...], dict[str, Any]]]:

    if not requests:
        return {}
    if len(requests) == 1:
        name, request = next(iter(requests.items()))
        return {
            name: cache_files(
                request.source_files,
                stage_root=stage_root,
                namespace=request.namespace,
            )
        }
    with ThreadPoolExecutor(
        max_workers=min(len(requests), 8),
        thread_name_prefix="uniaction-model-stage",
    ) as executor:
        futures = {
            name: executor.submit(
                cache_files,
                request.source_files,
                stage_root=stage_root,
                namespace=request.namespace,
            )
            for name, request in requests.items()
        }
        return {name: futures[name].result() for name in requests}
