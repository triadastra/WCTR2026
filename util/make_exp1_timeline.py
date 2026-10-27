#!/usr/bin/env python3
"""Exp 1 full-timeline plots (like Exp 2, but for the single aggregated national
series). For every aggregate model -- including the moving_average / random_walk /
seasonal_naive baselines -- plot the ENTIRE train->val->test history in real $,
with the 1-step test forecast overlaid and the splits shaded. Run from /root/full.
"""
import sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from src.data import TradeDataPipeline, DataConfig

DATA = "/root/data"; G = Path("graphs/exp1_aggregate"); AGG = Path("outputs/agg")

cfg = DataConfig(imports_dir=f"{DATA}/imports", exports_dir=f"{DATA}/exports",
                 input_len=36, lag_count=12, aggregate=True)
p = TradeDataPipeline(cfg); p.load_data(); p.create_splits()
g = p.df_full_grid.copy(); g["Time"] = pd.to_datetime(g["Time"]); g = g.sort_values("Time")
st = p.stats["exact"].iloc[0]
vmin, vmax = float(st["val_min"]), float(st["val_max"]); vr = (vmax - vmin) or 1.0
wmin, wmax = float(st["wt_min"]), float(st["wt_max"]); wr = (wmax - wmin) or 1.0
te, vs, ve, ts, tt = [pd.Timestamp(x) for x in (cfg.train_end, cfg.val_start, cfg.val_end, cfg.test_start, cfg.test_end)]
times = g["Time"].to_numpy()
CH = {"value": (g["Value"].to_numpy() * vr + vmin, vr, vmin, "pred_value"),
      "weight": (g["Weight"].to_numpy() * wr + wmin, wr, wmin, "pred_weight")}

STATIC = ["aggregate_value.png", "aggregate_weight.png", "residual_value.png", "loss_curve.png", "scatter_value.png"]

def billions(y, _):
    return f"${y/1e9:.1f}B"

def do(run):
    pc = AGG / run / "predictions.csv"
    if not pc.exists(): return False
    pr = pd.read_csv(pc, parse_dates=["Time"]).sort_values("Time")
    name = run.replace("_seed26", "")
    out = G / name; out.mkdir(parents=True, exist_ok=True)
    # carry over the static per-run figures too (so new models are complete)
    for fn in STATIC:
        src = AGG / run / "plots" / fn
        if src.exists(): shutil.copy(src, out / fn)
    for ch, (actual, rng, mn, col) in CH.items():
        if col not in pr.columns: continue
        pred = pr[col].to_numpy() * rng + mn
        fig, ax = plt.subplots(figsize=(13, 4.8))
        ax.axvspan(times.min(), te, color="#e9f3e9", alpha=.7)
        ax.axvspan(vs, ve, color="#fff1c9", alpha=.8); ax.axvspan(ts, tt, color="#fcdbd6", alpha=.8)
        ax.plot(times, actual, color="tab:blue", lw=1.4, label=f"actual aggregate {ch} (real $)")
        ax.plot(pr["Time"], pred, color="tab:red", lw=2.4, marker="o", ls="--", label="1-step forecast (test)")
        # keep the actual readable even if a baseline forecast is absurd: clamp the
        # view to the actual's range; off-scale forecast points are annotated.
        amax = float(np.nanmax(actual)); amin = float(np.nanmin(actual))
        pad = 0.08 * (amax - amin or 1.0)
        ax.set_ylim(max(0, amin - pad), amax + pad)
        pmax = float(np.nanmax(pred))
        if pmax > amax + pad:
            ax.annotate(f"forecast spikes off-scale (max ${pmax/1e9:.1f}B)\nactual high values not tracked",
                        (pr["Time"].iloc[int(np.nanargmax(pred))], amax + pad),
                        fontsize=8, va="top", ha="center", color="#c62828",
                        bbox=dict(boxstyle="round", fc="#fff0f0", ec="#c62828", alpha=.9))
        y1 = ax.get_ylim()[1]
        ax.annotate("train (fit + norm stats)", (times.min(), y1), fontsize=8, va="top", color="#2e7d32")
        ax.annotate("val", (vs, y1), fontsize=8, va="top", color="#b8860b")
        ax.annotate("test", (ts, y1), fontsize=8, va="top", color="#c62828")
        ax.set_title(f"Exp 1 aggregate — {name} — {ch}\nfull national timeline (real $, leakage-free: norm fit on train only)")
        ax.set_xlabel("Month"); ax.set_ylabel(f"Aggregate {ch} (real $)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(billions))
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=.25)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(out / f"timeline_{ch}.png", dpi=140); plt.close(fig)
    return True

runs = sorted({pp.parent.name for pp in AGG.glob("*_seed26/predictions.csv")})
for run in runs:
    print(f"[exp1-timeline] {run.replace('_seed26','')}: {'ok' if do(run) else 'no preds'}")
print(f"done: {len(runs)} runs")
