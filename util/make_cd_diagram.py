#!/usr/bin/env python
"""Equal-state sensitivity: Friedman–Nemenyi diagram of validated Exp 2 runs.
The family includes every canonical arm. A within-CD interval denotes failure
to reject a rank difference, never equivalence. Eight states and five shared
months do not provide a confirmatory independent-dataset benchmark.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "util"))

from ext.eval.resampling import friedman_nemenyi                    # noqa: E402
from make_significance import load_model_losses, state_loss_matrix          # noqa: E402

OUTDIR = ROOT / "graphs" / "_analysis"
TOP_N = 15

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
    "cross_attention": "flat CA", "film_attention": "FiLM",
    "none": "",
}
# Structure classes and their (CVD-validated) colors.
STRUCT_COLOR = {"flat": "#2a78d6", "axial": "#eb6834",
                "legacy": "#898781", "baseline": "#0b0b0b"}
STRUCT_OF = {
    "onehot": "flat", "embeddings": "flat",
    "cross_attention_2d": "axial", "cross_attention_3d": "axial",
    "cross_attention_4d": "axial",
    "2d": "axial", "2d_state": "axial", "3d": "axial", "4d": "axial",
    "cross_attention": "legacy", "film_attention": "legacy",
    "none": "baseline",
}


def split_arm(arm):
    for m in sorted(MODEL_DISPLAY, key=len, reverse=True):
        if arm == m or arm.startswith(m + "_"):
            return m, arm[len(m) + 1:] or "none"
    return arm, "none"


def label_of(arm):
    m, v = split_arm(arm)
    vd = VARIANT_DISPLAY.get(v, v)
    return f"{MODEL_DISPLAY.get(m, m)} {vd}".strip()


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_model_losses()
    if not data:
        print("[cd] no predictions under outputs/full/")
        return 1

    arms = sorted(data)
    X = state_loss_matrix(data, arms)
    res = friedman_nemenyi(X)
    ranks = pd.Series(res["mean_ranks"], index=arms).sort_values()
    cd = res["cd"]
    print(f"[cd] k={res['k']} arms, n={res['n']} states, "
          f"Friedman p={res['p']:.2e}, CD={cd:.3f}")
    tied = ranks[ranks <= ranks.iloc[0] + cd]
    print(f"[cd] {len(tied)} arms within CD of the best:")
    for a, r in tied.items():
        print(f"      {r:6.2f}  {label_of(a)}")

    means = {mv: float(np.mean(v[1])) for mv, v in data.items()}
    champ = min(means, key=means.get)
    champ_pos = int(ranks.index.get_loc(champ)) + 1
    first_md, first_md_pos = None, None
    for pos, (a, _) in enumerate(ranks.items(), start=1):
        _, v = split_arm(a)
        if STRUCT_OF.get(v) == "axial":
            first_md, first_md_pos = a, pos
            break

    make_figure(ranks, cd, res, champ, champ_pos, first_md, first_md_pos)
    return 0


def make_figure(ranks, cd, res, champ, champ_pos, first_md, first_md_pos):
    # PRINT FORM (2026-08-02): the classic Demšar layout (axis + crossing
    # leader lines) was unreadable at Procedia column width. Same content,
    # clearer geometry: one row per arm, dot at its mean rank, and the
    # Within-CD zone as a shaded band; this is not an equivalence interval.
    show = ranks.iloc[:TOP_N]
    best = float(show.iloc[0])
    edge = best + cd
    clique = show[show <= edge]

    fig, ax = plt.subplots(figsize=(8.8, 0.40 * TOP_N + 2.1))
    n_show = len(show)
    ys = np.arange(n_show)[::-1]
    lo = np.floor(best) - 0.15
    hi = max(float(show.max()) + 1.05, edge + .5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, TOP_N - 0.3 + 1.4)

    # The non-rejection interval around the rank reference.
    ax.axvspan(best, edge, color="#f2f0ea", zorder=0)
    ax.axvline(edge, color="#52514e", lw=1.4, linestyle=(0, (5, 3)), zorder=2)
    ax.text((best + edge) / 2, TOP_N + 0.62,
            f"within CD = {cd:.2f} of the leader — no rejection "
            f"(exploratory α = {res['alpha']})",
            ha="center", va="center", fontsize=11, color="#3a3a37",
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0), zorder=4)
    ax.annotate("", xy=(best, TOP_N + 0.15), xytext=(edge, TOP_N + 0.15),
                arrowprops=dict(arrowstyle="<->", color="#3a3a37", lw=1.4))

    for y, (arm, r) in zip(ys, show.items()):
        _, v = split_arm(arm)
        struct = STRUCT_OF.get(v, "baseline")
        col = STRUCT_COLOR[struct]
        tied = arm in clique.index
        ax.plot([lo, r], [y, y], color="#e1e0d9", lw=1.0, zorder=1)
        marker = "D" if struct == "baseline" else "o"
        ax.scatter([r], [y], s=130 if struct == "baseline" else 150,
                   color=col, marker=marker, zorder=3,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(f"{r:.2f}", (r, y), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="center",
                fontsize=11.5, color="#3a3a37",
                fontweight="bold" if tied else "normal")

    labels = [label_of(a) for a in show.index]
    ax.set_yticks(ys, labels, fontsize=12.5)
    for tick, arm in zip(ax.get_yticklabels(), show.index):
        _, v = split_arm(arm)
        tick.set_color(STRUCT_COLOR[STRUCT_OF.get(v, "baseline")])
        if arm in clique.index:
            tick.set_fontweight("bold")

    ax.set_xlabel(f"mean rank over {res['n']} state blocks "
                  f"(lower is better)", fontsize=12)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    handles = [plt.Line2D([], [], color=c, lw=0, marker="D" if s == "baseline"
                          else "o", markersize=9,
                          label={"flat": "flat (one-hot / embeddings)",
                                 "axial": "multidimensional (axial)",
                                 "legacy": "legacy combo encoder",
                                 "baseline": "naive baseline"}[s])
               for s, c in STRUCT_COLOR.items()]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=10.5, handletextpad=0.1,
              bbox_to_anchor=(0.995, 0.88))

    fig.suptitle("Critical-Difference Diagram on All Models",
                 fontsize=17, fontweight="bold", y=0.975)
    ax.set_title(
        f"All {res['k']} arms ranked within {res['n']} states (top {len(show)} shown)\n"
        "Non-rejection does not establish equivalence", fontsize=11, pad=14)
    fig.text(0.5, 0.036,
             f"Exploratory asymptotic Friedman p = {res['p']:.3g}; only {res['n']} state blocks.",
             ha="center", fontsize=9.5)
    fig.text(0.5, 0.014,
             f"Lowest pooled MSE: {label_of(champ)}; state mean-rank position {champ_pos}/{res['k']}.",
             ha="center", fontsize=9.5)

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    out = OUTDIR / "cd_diagram.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[cd] wrote {out.name}")


if __name__ == "__main__":
    raise SystemExit(main())
