#!/usr/bin/env python3
"""Build or verify the canonical Census source-file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.census_scope import (  # noqa: E402
    CENSUS_STATE_ITEMS,
    HS2_CODES,
    HS2_UNIVERSE_SHA256,
)


# This utility describes a fresh strict Census download, not the processed
# monthly benchmark. Never overwrite its active manifest by default.
DEFAULT_OUTPUT = ROOT / "config" / "census_data_manifest.json"
DEFAULT_RECEIPT = ROOT / "data" / "census_usa_trade_online_receipt.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def build_manifest(data_dir: Path, receipt_path: Path) -> dict[str, Any]:
    """Hash exactly one Census CSV per flow/state and bind its receipt."""
    data_dir = Path(data_dir).resolve()
    receipt_path = Path(receipt_path).resolve()
    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Census receipt {receipt_path}: {exc}") from exc
    scope = receipt.get("scope") if isinstance(receipt, dict) else None
    expected_states = [
        {"code": code, "name": name} for code, name in CENSUS_STATE_ITEMS
    ]
    if not isinstance(scope, dict) or scope.get("states") != expected_states:
        raise ValueError("Census receipt does not contain the frozen eight-state scope")
    if scope.get("hs2_codes") != list(HS2_CODES):
        raise ValueError("Census receipt does not contain the frozen HS2 scope")
    if scope.get("source_start") != "2008-01-01" or scope.get("source_end") != "2025-05-01":
        raise ValueError("Census receipt does not contain the frozen source calendar")

    expected_names = {f"{name}.csv" for _, name in CENSUS_STATE_ITEMS}
    files: list[dict[str, Any]] = []
    for flow in ("imports", "exports"):
        folder = data_dir / "data" / flow
        actual_names = {path.name for path in folder.glob("*.csv") if path.is_file()}
        if actual_names != expected_names:
            raise ValueError(
                f"{folder} must contain exactly {sorted(expected_names)}; "
                f"got {sorted(actual_names)}"
            )
        for name in sorted(actual_names):
            payload = (folder / name).read_bytes()
            files.append({
                "flow": flow,
                "name": name,
                "bytes": len(payload),
                "sha256": _sha256(payload),
            })

    # The loader uses compact separators; calculate that exact contract here.
    compact = json.dumps(
        {"version": 1, "files": files},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "fingerprint_algorithm": "sha256-manifest-v1",
        "source_fingerprint": _sha256(compact),
        "source_receipt": {
            "path": "data/census_usa_trade_online_receipt.json",
            "bytes": len(receipt_bytes),
            "sha256": _sha256(receipt_bytes),
        },
        "publisher": "U.S. Census Bureau",
        "product": "USA Trade Online",
        "states": sorted(name for _, name in CENSUS_STATE_ITEMS),
        "source_start": "2008-01-01",
        "source_end": "2025-05-01",
        "target_definition": {
            "commodity_universe": "all_hs_chapters",
            "hs_level": 2,
            "hs2_codes": list(HS2_CODES),
            "hs2_universe_sha256": HS2_UNIVERSE_SHA256,
            "country": "World Total",
            "measure": "total_vessel",
            "value_unit": "USD",
            "weight_unit": "kg",
        },
        "experiment_calendar": {
            "grid_start": "2009-01-01",
            "grid_end": "2025-05-01",
            "train_end": "2023-12-01",
            "val_start": "2024-01-01",
            "val_end": "2024-12-01",
            "test_start": "2025-01-01",
            "test_end": "2025-05-01",
        },
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = _canonical_json(build_manifest(args.data_dir, args.receipt))
        parsed = json.loads(payload)
        if args.verify:
            if args.output.read_bytes() != payload:
                raise ValueError(f"data manifest differs from source bytes: {args.output}")
            action = "Verified"
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
            try:
                temporary.write_bytes(payload)
                os.replace(temporary, args.output)
            finally:
                temporary.unlink(missing_ok=True)
            action = "Wrote"
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"{action} {args.output}: {len(parsed['files'])} files, "
        f"sha256:{parsed['source_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
