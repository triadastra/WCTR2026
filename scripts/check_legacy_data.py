#!/usr/bin/env python3
"""Read-only preflight for the configured, pinned source snapshot."""
import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloader import DataConfig, TradeDataPipeline, KEYS
from src.utils.config import get_data_config, load_config
from src.utils.experiment_manifest import exp1_source_domain, load_manifest
from src.utils.provenance import CORRECTED_NUM_COMBOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    # Audit the preserved source using the canonical train-only population.
    # Historical cohort selection is a separate reproduction workflow, not a
    # prerequisite for launching corrected experiments.
    cfg = load_config(str(ROOT / "config/base.yaml"))
    data = get_data_config(cfg)
    for flow in ("imports", "exports"):
        data[f"{flow}_dir"] = str(args.data_dir.resolve() / "data" / flow)
    from src.utils.config import resolve_data_paths
    data = resolve_data_paths(data, args.data_dir.resolve())
    pipe = TradeDataPipeline(DataConfig(**data))
    pipe.load_data()
    panel = pipe.filter_sparse_combos(pipe.build_grid(pipe.df_full))
    if len(pipe.kept_combos) != CORRECTED_NUM_COMBOS or panel.duplicated(KEYS + ["Time"]).any():
        raise RuntimeError("Canonical cohort/monthly-key regression")
    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    if pipe.source_fingerprint != manifest["PAPER_SOURCE_FINGERPRINT"]:
        raise RuntimeError("Source fingerprint differs from experiment manifest")
    if pipe.source_domain != exp1_source_domain(manifest):
        raise RuntimeError("Source domain/duplicate audit differs from experiment manifest")
    print(json.dumps({"source_fingerprint": pipe.source_fingerprint,
                      "source_domain": pipe.source_domain,
                      "retained_series": len(pipe.kept_combos),
                      "raw_files_modified": False}, indent=2))


if __name__ == "__main__":
    main()
