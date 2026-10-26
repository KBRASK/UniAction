from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


MIRROR_FORMAT = "uniaction-bf16-weight-mirror-v1"
_STAT_KEYS = ("device", "inode", "size", "mtime_ns", "ctime_ns")


def stat_receipt(path: Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _validate_receipt(path: Path, payload: dict[str, Any], *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = stat_receipt(path)
    expected = payload.get("stat")
    if not isinstance(expected, dict):
        raise ValueError(f"{label} has no immutable stat receipt")
    mismatch = [key for key in _STAT_KEYS if int(expected.get(key, -1)) != actual[key]]
    if mismatch:
        raise ValueError(
            f"{label} changed since BF16 materialization ({', '.join(mismatch)}); "
            "use the original DiT checkpoint instead of this cached copy"
        )


def validate_bf16_mirror(
    source_files: Iterable[Path],
    mirror_files: Iterable[Path],
    *,
    manifest_path: Path,
) -> dict[str, Any]:

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != MIRROR_FORMAT:
        raise ValueError(f"unsupported BF16 mirror format {payload.get('format')!r}")
    if payload.get("dtype") != "bfloat16" or payload.get("verified_exact") is not True:
        raise ValueError("BF16 mirror lacks an exact tensor verification receipt")

    source_files = tuple(Path(path) for path in source_files)
    mirror_files = tuple(Path(path) for path in mirror_files)
    source_entries = payload.get("source_files")
    mirror_entries = payload.get("mirror_files")
    if not isinstance(source_entries, list) or not isinstance(mirror_entries, list):
        raise ValueError("BF16 mirror manifest has invalid file inventories")
    if [path.name for path in source_files] != [entry.get("name") for entry in source_entries]:
        raise ValueError("BF16 mirror source inventory does not match the configured DiT")
    if [path.name for path in mirror_files] != [entry.get("name") for entry in mirror_entries]:
        raise ValueError("BF16 mirror file inventory does not match its manifest")
    for path, entry in zip(source_files, source_entries, strict=True):
        _validate_receipt(path, entry, label=f"BF16 source {path.name}")
    for path, entry in zip(mirror_files, mirror_entries, strict=True):
        _validate_receipt(path, entry, label=f"BF16 mirror {path.name}")

    return {
        "format": MIRROR_FORMAT,
        "manifest": str(manifest_path),
        "source_bytes": sum(int(entry["stat"]["size"]) for entry in source_entries),
        "mirror_bytes": sum(int(entry["stat"]["size"]) for entry in mirror_entries),
        "tensor_count": int(payload["tensor_count"]),
        "verified_exact": True,
    }
