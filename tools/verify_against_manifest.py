"""Check files against results/2026-10-26/manifest.json (SHA-256 and size).

Usage (from the repository root):
  python tools/verify_against_manifest.py                    # this repository's files ("path" field)
  python tools/verify_against_manifest.py --source-root DIR  # a copy laid out like the manifest's "source"
                                                             # paths, e.g. DIR/box/autodl-fs/wctr-exp1-numeric/...
Exits non-zero if any present file differs. Absent files are listed but are not errors:
the public release omits the two workbooks, and a partial copy omits whatever it did not fetch.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=ROOT / "results/2026-10-26/manifest.json", type=Path)
    ap.add_argument("--source-root", type=Path, help="check a copy addressed by the manifest's source paths")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    counts, absent, different = Counter(), [], []
    for entry in manifest["files"]:
        rel = entry["source"] if args.source_root else entry["path"]
        path = (args.source_root or ROOT) / rel
        if not path.is_file():
            counts["absent"] += 1
            absent.append(rel)
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            counts["DIFFERENT"] += 1
            different.append(rel)
        else:
            counts["match"] += 1

    print(f"{manifest.get('snapshot')}: {len(manifest['files'])} entries -> "
          f"{counts['match']} match, {counts['DIFFERENT']} different, {counts['absent']} absent")
    for rel in different:
        print("  DIFFERENT", rel)
    for rel in absent[:20]:
        print("  absent   ", rel)
    if len(absent) > 20:
        print(f"  ... {len(absent) - 20} more absent")
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(main())
