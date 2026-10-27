#!/usr/bin/env python3
"""Build or verify the frozen Experiment 1 monthly ground-truth dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import DataConfig  # noqa: E402
from src.data.ground_truth import (  # noqa: E402
    build_ground_truth_bundle,
    canonical_manifest_bytes,
    verify_artifacts,
    write_artifacts,
)
from src.utils.config import get_data_config, load_config  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "ground_truth" / "exp1_monthly_targets.csv.gz"
DEFAULT_MANIFEST = ROOT / "data" / "ground_truth" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "base.yaml",
        help="Base experiment config containing the frozen data contract.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT,
        help="Base directory containing the config's data/imports and data/exports.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--verify", action="store_true",
        help="Rebuild in memory and fail instead of writing if artifacts differ.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(str(args.config))
    data = get_data_config(cfg)
    base_dir = args.data_dir.resolve()
    for field in ("imports_dir", "exports_dir", "aggregated_csv"):
        if not data.get(field):
            continue
        configured = Path(data[field])
        data[field] = str(configured if configured.is_absolute() else base_dir / configured)

    config = DataConfig(**data)
    try:
        csv_bytes, gzip_bytes, manifest = build_ground_truth_bundle(
            config, artifact_name=args.output.name
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    manifest_bytes = canonical_manifest_bytes(manifest)

    if args.verify:
        verify_artifacts(
            args.output, args.manifest, gzip_bytes, manifest_bytes
        )
        action = "Verified"
    else:
        write_artifacts(
            args.output, args.manifest, gzip_bytes, manifest_bytes
        )
        action = "Wrote"

    artifact = manifest["artifact"]
    scope = manifest["scope"]
    print(
        f"{action} {args.output}: {scope['rows']:,} rows, "
        f"{scope['series']:,} series, {scope['months']} months, "
        f"sha256:{artifact['sha256']}"
    )
    print(
        f"Canonical CSV sha256:{artifact['canonical_csv_sha256']} "
        f"({len(csv_bytes):,} uncompressed bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
