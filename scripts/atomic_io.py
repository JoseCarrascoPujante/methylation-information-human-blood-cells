"""Durable publication of complete output directories, without replacing inputs."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np

from array_io import file_record, load_cache, prepare_cache, sha256

CACHE_PAYLOADS = [
    "discovery_beta.npy", "external_common_beta.npy", "common_discovery_indices.npy",
    "discovery_probe_ids.csv.gz", "common_probe_ids.csv.gz",
    "external_eligible_probe_ids.csv.gz", "discovery_samples.csv", "external_samples.csv",
]


def work_directory_name(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex[:12])


def fsync_directory(path: Path) -> None:
    # Windows cannot open a directory with os.open; this flush is a POSIX durability step.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(directory).parts):
            # Windows requires write access for fsync.
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
    for path in [*sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True), directory]:
        fsync_directory(path)


def publish_directory(working: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an existing output: {destination}")
    sync_tree(working)
    os.replace(working, destination)
    fsync_directory(destination.parent)


def verify_complete_cache(cache_dir: Path, protocol_path: Path | None = None) -> dict:
    cache = load_cache(cache_dir, protocol_path)
    checked_values = {}
    for key in ("discovery_beta", "external_beta"):
        matrix = cache[key]
        count = 0
        for start in range(0, matrix.shape[0], 20000):
            block = matrix[start:start + 20000]
            if not np.isfinite(block).all() or np.any((block < 0.0) | (block > 1.0)):
                raise ValueError(f"Invalid values during full cache read: {key}")
            count += int(block.size)
        checked_values[key] = count
    return {
        "status": "verified", "cache_manifest_sha256": sha256(cache_dir / "cache_manifest.json"),
        "full_beta_value_counts": checked_values,
        "files": {name: file_record(cache_dir / name) for name in cache["manifest"]["files"]},
        "discovery_shape": list(cache["discovery_beta"].shape),
        "external_common_shape": list(cache["external_beta"].shape),
    }


def finalize_cache_manifest(directory: Path) -> None:
    """Record only the eight scientific payloads, excluding runtime temporary files."""
    manifest_path = directory / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {name: file_record(directory / name) for name in CACHE_PAYLOADS}
    manifest["publication_code_sha256"] = sha256(Path(__file__))
    temporary = directory / "cache_manifest.pending.json"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)


def prepare_cache_atomic(
    discovery_path, external_path, discovery_metadata, external_metadata,
    destination, protocol, protocol_path
):
    working = work_directory_name(destination)
    prepare_cache(discovery_path, external_path, discovery_metadata, external_metadata,
                  working, protocol, protocol_path)
    finalize_cache_manifest(working)
    sync_tree(working)
    verify_complete_cache(working, protocol_path)
    publish_directory(working, destination.resolve())
    return verify_complete_cache(destination.resolve(), protocol_path)
