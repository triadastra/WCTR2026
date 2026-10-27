#!/usr/bin/env python
"""Descriptive ranks under MSE, MAE and MASE from validated Exp 2 artifacts.
Rank correlations and positions are computed from the current complete roster.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_paper_results import _exp2_rows, _aggregate_seeds
OUTDIR = ROOT / "graphs" / "_analysis"

FAMILY = {
    "lstm": "RNN", "gru": "RNN",
    "transformer": "Attention", "gpt": "Attention",
    "mamba": "SSM", "s4": "SSM", "s4_bidir": "SSM", "mamba_nd": "SSM", "s4nd": "SSM",
    "xgboost": "Tree", "lightgbm": "Tree", "random_forest": "Tree",
    "moving_average": "Baseline", "random_walk": "Baseline",
    "seasonal_naive": "Baseline",
}
FAM_COLOR = {"Tree": "#eb6834", "SSM": "#2a78d6", "Attention": "#1baf7a",
             "RNN": "#898781", "Baseline": "#0b0b0b"}
MODEL_DISPLAY = {
    "lstm": "LSTM", "gru": "GRU", "transformer": "Transformer", "gpt": "GPT",
    "mamba": "Mamba", "s4": "S4", "s4_bidir": "S4-Bidirectional", "mamba_nd": "Mamba-ND", "s4nd": "S4ND",
    "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "moving_average": "Moving average", "random_walk": "Random walk",
    "seasonal_naive": "Seasonal naive",
}
VARIANT_DISPLAY = {
    "onehot": "one-hot", "embeddings": "embed",
    "cross_attention_2d": "axial 2D", "cross_attention_3d": "axial 3D",
    "cross_attention_4d": "axial 4D",
    "2d": "2D", "2d_state": "2D-state", "3d": "3D", "4d": "4D",
    "cross_attention": "flat CA", "film_attention": "FiLM", "none": "",
}


def split_arm(arm):
    for m in sorted(FAMILY, key=len, reverse=True):
        if arm == m or arm.startswith(m + "_"):
            return m, arm[len(m) + 1:] or "none"
    return arm, "none"


def label_of(arm):
    m, v = split_arm(arm)
    return f"{MODEL_DISPLAY.get(m, m)} {VARIANT_DISPLAY.get(v, v)}".strip()


def load():
    t = _aggregate_seeds(_exp2_rows(ROOT))
    t["arm"] = t.model.astype(str) + "_" + t.variant.astype(str)
    for col, rc in (("norm_mse_combined", "r_mse"), ("mae", "r_mae"),
                    ("mase", "r_mase")):
        t[rc] = t[col].rank(method="min").astype(int)
    t["family"] = [FAMILY.get(split_arm(a)[0], "?") for a in t.arm]
    return t.reset_index(drop=True)


def nudge(items, gap=1.0):
    """items: [(rank, text, color)] -> y positions >= gap apart, in order."""
    items = sorted(items, key=lambda it: it[0])
    ys = []
    for r, *_ in items:
        y = r if not ys else max(r, ys[-1] + gap)
        ys.append(y)
    return [(y, it) for y, it in zip(ys, items)]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t = load()
    n = len(t)
    rho = {}
    for a, b in (("r_mse", "r_mae"), ("r_mse", "r_mase"), ("r_mae", "r_mase")):
        rho[(a, b)] = float(np.corrcoef(t[a], t[b])[0, 1])
    print(f"[slope] {n} arms | Spearman MSE~MAE {rho[('r_mse','r_mae')]:+.2f}"
          f"  MSE~MASE {rho[('r_mse','r_mase')]:+.2f}"
          f"  MAE~MASE {rho[('r_mae','r_mase')]:+.2f}")

    # Who gets direct labels: top 5 on the outer columns, all trees, MA.
    picked = set(t.nsmallest(5, "r_mse").arm) | set(t.nsmallest(5, "r_mase").arm)
    picked |= set(t[t.family.isin(["Tree", "Baseline"])].arm)

    fig, ax = plt.subplots(figsize=(9.8, 0.155 * n + 2.9))
    ax.set_xlim(-1.55, 3.55)
    ax.set_ylim(n + 1.2, -4.2)          # rank 1 at the top; headroom for headers
    ax.axis("off")

    cols = [("r_mse", 0, "MSE\n(pooled)"), ("r_mae", 1, "MAE\n(pooled)"),
            ("r_mase", 2, "MASE\n(per-series)")]
    for _, x, name in cols:
        ax.axvline(x, color="#e1e0d9", lw=1.2, zorder=0)
        ax.text(x, -1.1, name, ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    # Draw non-tree families first, trees on top so the movers stay visible.
    order = ["RNN", "Attention", "SSM", "Baseline", "Tree"]
    for fam in order:
        for _, r in t[t.family == fam].iterrows():
            xs, ys = [0, 1, 2], [r.r_mse, r.r_mae, r.r_mase]
            dash = (0, (4, 2.4)) if fam == "Baseline" else "solid"
            lw = 2.4 if fam == "Tree" else (1.6 if fam == "Baseline" else 1.3)
            al = 0.95 if fam in ("Tree", "Baseline") else 0.45
            ax.plot(xs, ys, color=FAM_COLOR[fam], lw=lw, alpha=al,
                    linestyle=dash, solid_capstyle="round", zorder=3)
            if r.arm in picked:
                ax.scatter(xs, ys, s=14, color=FAM_COLOR[fam], zorder=4)

    # Direct labels on the outer columns, nudged apart where ranks collide.
    lab = t[t.arm.isin(picked)]
    left = [(r.r_mse, f"{label_of(r.arm)}  {r.r_mse}",
             FAM_COLOR[r.family]) for _, r in lab.iterrows()]
    right = [(r.r_mase, f"{r.r_mase}  {label_of(r.arm)}",
              FAM_COLOR[r.family]) for _, r in lab.iterrows()]
    for y, (rk, txt, col) in nudge(left):
        ax.text(-0.12, y, txt, ha="right", va="center", fontsize=9, color=col)
        if abs(y - rk) > 0.01:
            ax.plot([-0.10, -0.02], [y, rk], color="#c3c2b7", lw=0.7)
    for y, (rk, txt, col) in nudge(right):
        ax.text(2.12, y, txt, ha="left", va="center", fontsize=9, color=col)
        if abs(y - rk) > 0.01:
            ax.plot([2.02, 2.10], [rk, y], color="#c3c2b7", lw=0.7)

    handles = [plt.Line2D([], [], color=FAM_COLOR[f], lw=3,
                          linestyle="--" if f == "Baseline" else "-", label=f)
               for f in ("Tree", "SSM", "Attention", "RNN", "Baseline")]
    ax.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
              fontsize=9.5, bbox_to_anchor=(0.5, -0.045))

    fig.suptitle("Model Ranking Under MSE, MAE, and MASE",
                 fontsize=15, fontweight="bold", y=0.98)
    ax.set_title(
        f"rank of each of the {n} arms (1 = best) — identical predictions, "
        f"different error metric\n"
        f"Spearman: MSE~MAE {rho[('r_mse','r_mae')]:+.2f} · "
        f"MAE~MASE {rho[('r_mae','r_mase')]:+.2f}",
        fontsize=10, color="#333333", pad=14)
    fig.text(0.5, 0.033,
             "MSE and MAE pooled over 4,590 series-months; MASE per-series "
             "(seasonal-naive m=12), averaged over 918 series. "
             "Source: validated run artifacts.",
             ha="center", fontsize=8.5, color="#555555")
    fig.text(0.5, 0.014,
             "Ranks are descriptive; they do not establish significance or equivalence.",
             ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.95))
    out = OUTDIR / "rank_slopegraph.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[slope] wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
