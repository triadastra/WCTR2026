#!/usr/bin/env python
"""Simultaneous axial-vs-flat tests within each backbone.

Every axial arm is compared with both predeclared flat encodings (onehot and
embeddings). No flat comparator is selected by test MSE. Units are the 918
lanes; the state is the cluster (cluster-robust paired t, wild cluster
bootstrap p; see ext/eval/resampling.py). BH-FDR is applied across the
complete comparison family.

Usage:  python util/make_axial_vs_flat.py
"""
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "util"))

from ext.eval.resampling import holm, benjamini_hochberg  # noqa: E402
from make_significance import load_model_losses, paired_comparison    # noqa: E402

OUTDIR = ROOT / "graphs" / "_analysis"
BACKBONES = ("lstm", "gru", "transformer", "gpt", "mamba", "s4")
DIMS = ("2d", "3d", "4d")


def main():
    data = load_model_losses()
    combo = {mv: v[0] for mv, v in data.items()}
    rows = []
    for b in BACKBONES:
        for flat_variant in ("onehot", "embeddings"):
            flat = f"{b}_{flat_variant}"
            if flat not in combo:
                raise RuntimeError(f"Missing required arm {flat}")
            for dim in DIMS:
                ax = f"{b}_cross_attention_{dim}"
                if ax not in combo:
                    raise RuntimeError(f"Missing required arm {ax}")
                result = paired_comparison(combo[ax], combo[flat])
                rows.append({**result, "backbone": b, "axial": dim,
                             "flat_arm": flat_variant,
                             "axial_worse": bool(result["mean_lane_delta"] > 0)})
    tbl = pd.DataFrame(rows)
    p = [1. if math.isnan(x) else x for x in tbl.p_wild_cluster]
    tbl["q_bh"] = benjamini_hochberg(p)
    tbl["p_holm"] = holm(p)
    tbl["verdict"] = ["degenerate" if d else "not significant" if q >= .05 else
                      ("axial worse" if w else "axial better")
                      for q, w, d in zip(tbl.q_bh, tbl.axial_worse, tbl.degenerate)]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(OUTDIR / "axial_vs_flat.csv", index=False)
    print(tbl.to_string(index=False))
    n_ind = (tbl.verdict == "not significant").sum()
    print(f"\n[axial-vs-flat] {len(tbl)} pairs: {n_ind} not significant, "
          f"{(tbl.verdict == 'axial worse').sum()} axial-worse, "
          f"{(tbl.verdict == 'axial better').sum()} axial-better -> "
          f"{OUTDIR / 'axial_vs_flat.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
