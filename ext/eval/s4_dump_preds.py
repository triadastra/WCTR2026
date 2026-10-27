#!/usr/bin/env python3
"""ADDITIVE: dump per-series S4/S4ND predictions to CSV for the DM test.

Reuses scripts/s4_eval.py BY IMPORT -- it does NOT modify it. The trick:
s4_eval.evaluate() calls `_aggregate_plots(ds, pred, targ, run_dir, exp)` with
exactly the arrays we need, so we monkeypatch that one function to also write a
predictions.csv in the same schema the github models emit:

    State,Commodity,Import/Export,Time,pred_value,pred_weight,true_value,true_weight

One experiment per process (OmegaConf 'eval' resolver re-registers otherwise).
Outputs go under ext/outputs/s4_preds/<run_dir>/predictions.csv so that
outputs/full/ stays untouched.

Usage:
    python ext/eval/s4_dump_preds.py s4_embeddings   # from the repo root
"""
import os
from pathlib import Path
import sys
import csv

# See build_results_xlsx.py: resolve the repo from this file, not a hardcoded
# box path. Override with the REPO environment variable.
REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[2]))
S4 = os.path.join(REPO, "external", "s4")
OUT_ROOT = os.path.join(REPO, "ext", "outputs", "s4_preds")

# s4_eval expects cwd == external/s4 (hydra ./configs, `import train`, `import src`)
os.chdir(S4)
sys.path.insert(0, S4)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import s4_eval as se  # noqa: E402


def _dump_predictions(ds, pred, targ, run_dir, exp):
    """Replacement for se._aggregate_plots: write predictions.csv (normalized)."""
    keys, times = ds.test_keys, ds.test_times
    if len(keys) != pred.shape[0]:
        print(f"[dump-warn] {exp}: key/pred misalign {len(keys)} != {pred.shape[0]}")
        return
    run_name = os.path.basename(str(run_dir))
    out_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.csv")
    n_written = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["State", "Commodity", "Import/Export", "Time",
                    "pred_value", "pred_weight", "true_value", "true_weight"])
        for k in range(pred.shape[0]):
            key = keys[k]
            # canonical github combo key is (State, Commodity, Import/Export)
            if isinstance(key, (tuple, list)) and len(key) == 3:
                st, cm, fl = key
            else:
                print(f"[dump-warn] {exp}: unexpected key shape {key!r}")
                return
            t = str(times[k])[:10]
            w.writerow([st, cm, fl, t,
                        float(pred[k, 0]), float(pred[k, 1]),
                        float(targ[k, 0]), float(targ[k, 1])])
            n_written += 1
    print(f"[dump-ok] {exp}: wrote {n_written} rows -> {path}")
    # sanity: show one key so we can confirm column order matches github
    print(f"[dump-key] {exp}: first key = {keys[0]!r}  time = {str(times[0])[:10]}")


def main():
    if len(sys.argv) < 2:
        print("usage: s4_dump_preds.py <experiment>")
        print("experiments:", list(se.EXPERIMENTS))
        sys.exit(1)
    exp = sys.argv[1]
    if exp not in se.EXPERIMENTS:
        print(f"unknown experiment {exp}; choices: {list(se.EXPERIMENTS)}")
        sys.exit(1)
    se._aggregate_plots = _dump_predictions  # monkeypatch -- no s4_eval edits
    se.evaluate(exp, se.EXPERIMENTS[exp])


if __name__ == "__main__":
    main()
