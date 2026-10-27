"""Durable local run records, also importable from the vendored S4 namespace."""
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def sync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_event(path, event, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**fields, "event": event, "timestamp": utc_now()}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_finite_json(row), default=str, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_checkpoint_manifest(root, files, *, kind, run_contract):
    root = Path(root).resolve()
    records = []
    for role, relative in files.items():
        path = (root / relative).resolve()
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size == 0:
            raise ValueError(f"Empty checkpoint artifact: {path}")
        records.append({"role": role, "path": relative, "bytes": size,
                        "sha256": digest.hexdigest()})
    if not records:
        raise ValueError("A checkpoint manifest requires at least one artifact")
    payload = {"version": 1, "kind": kind, "created_at": utc_now(),
               "run_contract": run_contract, "files": records}
    atomic_text(root / "checkpoint_manifest.json",
                json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload
