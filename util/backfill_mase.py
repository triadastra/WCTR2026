#!/usr/bin/env python
"""Backfill MASE/RMSSE into run metrics that are missing them.

WHY
---
9 of 48 Exp 2 arms carry NaN MASE/RMSSE in results.xlsx, and they are not a
random 9 -- they include `s4 embeddings`, which is RANK 2 and the flat arm the
headline "axial is indistinguishable from flat" claim is measured against. A
MASE table published as-is would be blank exactly where the comparison lives.

Two separate causes:
  * The S4 family (s4 embeddings/onehot, s4nd 2d/2d_state/3d/4d) is scored by
    scripts/s4_eval.py, which never computes MASE at all.
  * The three naive baselines DO call add_mase_rmsse via run_baseline.py:121,
    but it took its silent path: on `if mav and maw` being empty it sets no
    keys and prints nothing, so the failure left no trace in any log.

This script recomputes them from each run's saved predictions, which is
possible entirely offline -- no GPU, no checkpoints (none are tracked;
.gitignore:44 excludes *.pth).

CORRECTNESS
-----------
It calls scripts/inference.py::add_mase_rmsse directly rather than reimplementing
the metric. That matters twice over:

  1. The convention is M4/M5 -- scale EACH series by its own in-sample
     seasonal-naive (m=12) error, then average across series, NOT a pooled
     ratio. A hand-rolled version would almost certainly pool and silently
     disagree with the 39 arms that already have values.
  2. Units. The scale comes from pipeline.df_full_grid, which dataloader.py
     assigns at :494 AFTER normalizing at :479 -- so it is normalized, matching
     the normalized prediction errors. Reading predictions.csv instead of
     predictions_norm.csv for the S4 family would divide real-dollar errors by
     a 0-1 scale. That exact real-units-vs-normalized trap has already produced
     three separate defects in this project (compact-workbook sMAPE,
     Exp2_Ensemble, and the weight-decay ablation), so the file choice below is
     deliberate, not incidental.

Usage:
    python util/backfill_mase.py            # backfill runs missing MASE
    python util/backfill_mase.py --verify   # ALSO recompute arms that already
                                            # have it and report the delta
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.exp2_artifacts import validated_exp2_artifacts  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))

from src.utils import load_config                      # noqa: E402
from src.data import TradeDataPipeline, DataConfig     # noqa: E402
from inference import add_mase_rmsse                   # noqa: E402

RUNS = ROOT / "outputs" / "full"
KEY = ["State", "Commodity", "Import/Export"]


def build_pipeline(config="config/base.yaml"):
    """Normalized full grid + train_end, the two things the metric needs."""
    cfg = load_config(str(ROOT / config))
    fields = DataConfig.__dataclass_fields__
    dc = DataConfig(**{k: v for k, v in cfg["data"].items() if k in fields})
    # Paths in base.yaml are relative to the repo root.
    for attr in ("imports_dir", "exports_dir"):
        p = getattr(dc, attr, None)
        if p and not Path(p).is_absolute():
            setattr(dc, attr, str(ROOT / p))
    p = TradeDataPipeline(dc)
    p.load_data()
    p.create_splits()
    return p


def load_predictions(run_dir: Path):
    """Prefer predictions_norm.csv -- see the units note in the module docstring."""
    norm = run_dir / "predictions_norm.csv"
    plain = run_dir / "predictions.csv"
    src = norm if norm.exists() else plain
    if not src.exists():
        return None, None, None, None
    df = pd.read_csv(src)
    need = {"pred_value", "pred_weight", "true_value", "true_weight", *KEY}
    if not need.issubset(df.columns):
        return None, None, None, None
    pred = df[["pred_value", "pred_weight"]].to_numpy(dtype=float)
    targ = df[["true_value", "true_weight"]].to_numpy(dtype=float)
    keys = [tuple(str(x) for x in row) for row in df[KEY].to_numpy()]
    return pred, targ, keys, src.name


def main():
    # Provenance gate: never publish a MASE backfill from artifacts evaluated on
    # the pre-correction, survivorship-biased cohort.
    validated_exp2_artifacts(ROOT, RUNS)
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="Also recompute arms that already have MASE and report the delta.")
    ap.add_argument("--dry-run", action="store_true", help="Do not write metrics.json.")
    args = ap.parse_args()

    if not RUNS.is_dir():
        print(f"[mase] runs dir not found: {RUNS}")
        return 1

    print("[mase] building pipeline (normalized full grid)...")
    pipeline = build_pipeline()

    filled, verified, skipped, failed = [], [], [], []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        payload = json.loads(mj.read_text())
        metrics = payload.get("metrics", payload)
        has = "mase" in metrics and metrics["mase"] is not None and not (
            isinstance(metrics["mase"], float) and np.isnan(metrics["mase"]))
        if has and not args.verify:
            skipped.append(d.name)
            continue

        pred, targ, keys, srcname = load_predictions(d)
        if pred is None:
            failed.append((d.name, "no usable predictions file"))
            continue

        fresh = {}
        add_mase_rmsse(fresh, pipeline, pred, targ, keys)
        if "mase" not in fresh:
            failed.append((d.name, "add_mase_rmsse produced nothing (no key overlap?)"))
            continue

        if has:
            delta = abs(fresh["mase"] - float(metrics["mase"]))
            verified.append((d.name, float(metrics["mase"]), fresh["mase"], delta))
            continue

        metrics.update({k: float(v) for k, v in fresh.items()})
        if "metrics" in payload:
            payload["metrics"] = metrics
        else:
            payload = metrics
        if not args.dry_run:
            mj.write_text(json.dumps(payload, indent=2))
        filled.append((d.name, srcname, fresh["mase"]))

    print(f"\n[mase] filled {len(filled)} run(s):")
    for name, src, v in filled:
        print(f"    {name:44s} mase={v:.6f}  (from {src})")
    if verified:
        worst = max(verified, key=lambda r: r[3])
        print(f"\n[mase] verified {len(verified)} arm(s) that already had MASE; "
              f"max |delta| = {worst[3]:.2e} on {worst[0]}")
        if worst[3] > 1e-6:
            print("[mase] WARNING: recomputation disagrees with stored values -- "
                  "the conventions have diverged, do NOT trust the backfill")
    if failed:
        print(f"\n[mase] {len(failed)} failure(s):")
        for name, why in failed:
            print(f"    {name:44s} {why}")
    print(f"\n[mase] {len(skipped)} run(s) already had MASE and were left alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
