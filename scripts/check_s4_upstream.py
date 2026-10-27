#!/usr/bin/env python
"""Verify the pinned S4 source audit, optionally against an authors' checkout.

This is read-only: it never downloads, replaces, or repairs vendor files.
The manifest distinguishes unchanged upstream code from disclosed local patches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify(vendor, upstream=None):
    vendor = Path(vendor)
    manifest = json.loads((vendor / "UPSTREAM.json").read_text())
    errors = []
    if upstream is not None:
        upstream = Path(upstream)
        commit = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if commit != manifest["commit"]:
            errors.append(f"upstream HEAD {commit} differs from pinned {manifest['commit']}")
    counts = {}
    for relative, entry in manifest["files"].items():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        path = vendor / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if digest != entry["local_sha256"]:
            errors.append(f"local source changed since audit: {relative}")
        if upstream is not None:
            path = upstream / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if digest != entry["upstream_sha256"]:
                errors.append(f"upstream source differs from audit: {relative}")
    for relative in manifest["unmodified_s4_core"]:
        entry = manifest["files"][relative]
        if entry["status"] != "identical" or entry["local_sha256"] != entry["upstream_sha256"]:
            errors.append(f"S4 core is not unchanged upstream code: {relative}")
    # Added modules/configs can change imports even when every old file hashes
    # correctly (e.g. adding an __init__.py to a namespace package).
    for directory in ("src", "models", "configs"):
        for path in sorted((vendor / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                relative = path.relative_to(vendor).as_posix()
                if relative not in manifest["files"]:
                    errors.append(f"unaudited source/config file: {relative}")
    return {"repository": manifest["repository"], "commit": manifest["commit"],
            "files": counts, "unmodified_s4_core_files": len(manifest["unmodified_s4_core"]),
            "upstream_checkout_checked": upstream is not None, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-checkout", type=Path)
    args = parser.parse_args()
    report = verify(ROOT / "external/s4", args.upstream_checkout)
    print(json.dumps(report, indent=2))
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
