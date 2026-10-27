#!/usr/bin/env python
"""Collect metrics from a full-test run folder into one results.csv.

Scans <runs-dir>/<model>_<variant>_seed<seed>/metrics.json and writes
<runs-dir>/results.csv, one row per run. Requires the complete shared metric suite and canonical Experiment 2 recipes.
This per-seed diagnostic may be partial; the paper builder requires all seeds.

Usage:
    python scripts/collect_results.py --runs-dir outputs/full
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.provenance import (
    corrected_per_combo_artifact,
    corrected_prediction_artifact,
)
from src.inference.metrics import REQUIRED_EVALUATION_METRICS

RUN_DIR_RE = re.compile(r"^(?P<model>.+?)_(?P<variant>[^_]+)_seed(?P<seed>\d+)$")

# Normalized scale-dependent metrics and physical-unit percentages use the
# same canonical schema as the evaluator, provenance gate, and paper builder.
METRIC_COLS = list(REQUIRED_EVALUATION_METRICS)


def num_or_blank(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="outputs/full")
    p.add_argument("--out", default=None, help="Defaults to <runs-dir>/results.csv")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out) if args.out else runs_dir / "results.csv"

    rows = []
    from src.utils.exp2_protocol import validate_contract
    from src.utils.exp2_protocol import expected_runs
    from src.utils.experiment_manifest import load_manifest
    from src.utils.exp2_completion import validate_exp2_completion_artifact
    from src.utils.provenance import prediction_truth_digest
    expected = {r.directory_name: r for r in expected_runs(load_manifest(Path(__file__).resolve().parents[1] / "config/experiment_manifest.env"))}
    truth = None
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m:
            print(f"[warn] skipping non-standard dir name: {d.name}")
            continue
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            print(f"[warn] no metrics.json in {d.name} (incomplete run?)")
            continue
        with open(metrics_path) as f:
            payload = json.load(f)
        if type(payload.get("seed")) is not int or payload["seed"] != int(m.group("seed")):
            raise RuntimeError(f"Refusing directory/payload seed mismatch: {d.name}")
        run = expected.get(d.name)
        if run is None or any(payload.get(k) != getattr(run, k) for k in ("model", "variant", "seed")):
            raise RuntimeError(f"Refusing directory/payload model or variant mismatch: {d.name}")
        validate_exp2_completion_artifact(payload, metrics_path)
        current, reason = corrected_per_combo_artifact(payload)
        if not current:
            raise RuntimeError(
                f"Refusing stale run {d.name}: {reason}. Rerun it under the "
                "corrected train-only cohort."
            )
        current, reason = corrected_prediction_artifact(payload, metrics_path, verify_scaled_metrics=True)
        if not current:
            raise RuntimeError(
                f"Refusing incomplete run {d.name}: {reason}. Rerun its "
                "corrected evaluation."
            )
        current, reason = validate_contract(payload, Path(__file__).resolve().parents[1])
        if not current:
            raise RuntimeError(f"Refusing off-protocol run {d.name}: {reason}")
        digest = prediction_truth_digest(payload, metrics_path)
        if truth is not None and digest != truth:
            raise RuntimeError(f"Test targets differ across runs: {d.name}")
        truth = digest
        flat = payload.get("metrics", payload)
        # Prefer the model/variant recorded in the payload (handles
        # multi-underscore variants like cross_attention that the dir-name
        # regex would split wrong).
        row = {
            "model": payload.get("model", m.group("model")),
            "variant": payload.get("variant", m.group("variant")),
            "seed": payload["seed"],
            "mode": payload.get("mode", ""),
            "n_samples": payload.get("n_samples", ""),
        }
        for k in METRIC_COLS:
            row[k] = num_or_blank(flat.get(k))
        row["num_params"] = num_or_blank(flat.get("num_params"))
        # Two-sided over the input window or causal (None/blank for trees).
        # Recorded per run since 2026-09-09; blank for older artifacts.
        arch = payload.get("architecture") or {}
        row["bidirectional"] = ("" if arch.get("bidirectional") is None
                                else bool(arch["bidirectional"]))
        rows.append(row)

    if not rows:
        raise RuntimeError("No validated completed runs found; results.csv was not replaced")

    # Guard: every row must report norm_mse_combined on the SAME (normalized
    # 0-1) scale. A blank means that run was evaluated WITHOUT --normalized, so
    # its shared columns (mae/rmse/value_mse) are real USD/kg and the leaderboard
    # would silently mix scales. Surface this loudly instead of ranking nonsense.
    blank = [f"{r['model']}_{r['variant']}_seed{r['seed']}"
             for r in rows if r["norm_mse_combined"] == ""]
    modes = sorted({r["mode"] for r in rows if r["mode"]})
    if blank:
        print(f"[ERROR] {len(blank)} run(s) have a BLANK norm_mse_combined — evaluated "
              f"without --normalized, so the leaderboard would mix USD/kg with 0-1 scales:")
        print(f"        {blank}")
        print("        Re-run those with --normalized (inference.py / run_baseline.py) "
              "or s4_eval.py for S4 grid runs.")
    if len(modes) > 1:
        print(f"[WARN] runs span multiple eval modes {modes}; expected all 'one_step_normalized'.")

    # Sort by the common normalized metric (blanks last).
    def sort_key(r):
        v = r["norm_mse_combined"]
        return (v == "", v if v != "" else 0)
    rows.sort(key=sort_key)

    cols = ["model", "variant", "seed", "mode", "n_samples"] + METRIC_COLS + ["num_params", "bidirectional"]
    temporary = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    with open(temporary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(temporary, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
