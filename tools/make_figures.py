"""Draw the README figures from the recomputed tables.

Usage (from the repository root, after tools/recompute_results.py):
  python tools/make_figures.py [RECOMPUTED_DIR] [OUT_DIR]
Defaults: docs/paper_audit/recomputed  figures
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/paper_audit/recomputed"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Chart chrome (light chart surface; the PNGs carry their own background).
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#a3302f", "#e34948", "#f0efec", "#2a78d6", "#104281"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
})

full = pd.read_csv(REC / "fig3_table2_full_arms.csv", index_col=0)
MA = full.loc["moving_average|none"]


def text_ink(rgba):
    r, g, b = rgba[:3]
    return INK if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.55 else "#ffffff"


# ---------------------------------------------------------------- 1. ND improvement heatmap
ROWS = [("S4", "s4"), ("S4 (bidir.)", "s4_bidir"), ("Mamba-2", "mamba"), ("Mamba-ND", "mamba_nd"),
        ("S4ND", "s4nd"), ("Transformer", "transformer"), ("GPT", "gpt"), ("LSTM", "lstm"),
        ("GRU", "gru"), ("XGBoost", "xgboost"), ("LightGBM", "lightgbm"), ("Random Forest", "random_forest")]
COLS = [("Flat\none-hot", "onehot"), ("Flat\nembed.", "embeddings"), ("ASA\n1-D", "cross_attention"),
        ("FiLM", "film_attention"), ("2D", "cross_attention_2d"), ("3D", "cross_attention_3d"),
        ("4D", "cross_attention_4d"), ("2D +\ngroup axis", None)]
S4ND_VARIANT = {"cross_attention_2d": "2d_state", "cross_attention_3d": "3d", "cross_attention_4d": "4d"}


def arm_for(model, variant, col_label):
    if model == "s4nd":
        if col_label.startswith("2D +"):
            return "s4nd|2d"
        variant = S4ND_VARIANT.get(variant)
    if variant is None:
        return None
    arm = f"{model}|{variant}"
    return arm if arm in full.index else None


fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"wspace": 0.28})
for ax, metric, title in zip(axes, ("mse", "mae"), ("MSE (training objective)", "MAE")):
    grid = np.full((len(ROWS), len(COLS)), np.nan)
    dag = np.zeros_like(grid, dtype=bool)
    for i, (_, m) in enumerate(ROWS):
        for j, (clab, v) in enumerate(COLS):
            a = arm_for(m, v, clab)
            if a:
                grid[i, j] = full.loc[a, f"imp_{metric}"]
                dag[i, j] = bool(full.loc[a, f"dagger_{metric}"])
    lim = np.ceil(np.nanmax(np.abs(grid)))
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    ax.imshow(np.ma.masked_invalid(grid), cmap=DIVERGING, norm=norm, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            rgba = DIVERGING(norm(grid[i, j]))
            ax.text(j, i, f"{grid[i, j]:+.2f}" + ("†" if dag[i, j] else ""), ha="center", va="center",
                    fontsize=7.5, color=text_ink(rgba))
    ax.set_xticks(range(len(COLS)), [c[0] for c in COLS], fontsize=8)
    ax.set_yticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=8.5)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for y in (4.5, 6.5, 8.5):
        ax.axhline(y, color=INK2, linewidth=0.8)
    ax.set_title(f"{title}: % improvement over Moving Average", pad=34, loc="left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    cb.set_label("better than MA  →", color=INK2, fontsize=8)
fig.text(0.01, 0.015,
         "Mean of seeds 26/105/115 on 918 lanes × Jan–May 2025. † = beats Moving Average with uncorrected "
         "wild-cluster bootstrap p < 0.05 (8 state clusters); no arm survives Benjamini–Hochberg correction.\n"
         "S4ND: 2D = state × time (s4nd_2d_state); \"2D + group axis\" = state × commodity-flow group × time "
         "(s4nd_2d, the cell printed as \"2D\" in the paper's Fig. 3).", fontsize=7.5, color=INK2)
fig.savefig(OUT / "nd_improvement_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2. MSE rank vs MAE rank
FAMILY = {"s4": "SSM", "s4_bidir": "SSM", "mamba": "SSM", "mamba_nd": "SSM", "s4nd": "SSM",
          "transformer": "Attention", "gpt": "Attention", "lstm": "RNN", "gru": "RNN",
          "xgboost": "Tree", "lightgbm": "Tree", "random_forest": "Tree",
          "moving_average": "Baseline", "random_walk": "Baseline", "seasonal_naive": "Baseline"}
STYLE = {"SSM": (BLUE, "o"), "Attention": (ORANGE, "o"), "RNN": (AQUA, "o"),
         "Tree": (MUTED, "s"), "Baseline": (MUTED, "X")}
full["family"] = full.model.map(FAMILY)
fig, ax = plt.subplots(figsize=(7.2, 6.4))
n = len(full)
ax.plot([1, n], [1, n], color=GRID, linewidth=1, zorder=0)
ax.text(3.2, 4.4, "same rank on both", color=MUTED, fontsize=7, rotation=45, rotation_mode="anchor")
for fam, (col, mk) in STYLE.items():
    sub = full[full.family == fam]
    ax.scatter(sub.rank_mse, sub.rank_mae, s=58, c=col, marker=mk, edgecolors=SURF, linewidths=1.5,
               label=f"{fam} ({len(sub)})", zorder=3)
LABELS = {"s4|cross_attention_3d": "S4 ASA 3D", "lstm|embeddings": "LSTM embeddings",
          "transformer|film_attention": "Transformer FiLM", "xgboost|onehot": "XGBoost one-hot",
          "moving_average|none": "Moving Average"}
LEADER = {"transformer|film_attention": (27, 5.5), "lstm|embeddings": (27, 1.5)}  # empty band, data coords
for arm, lab in LABELS.items():
    x, y = full.loc[arm, "rank_mse"], full.loc[arm, "rank_mae"]
    if arm in LEADER:
        ax.annotate(f"{lab} ({int(x)}, {int(y)})", (x, y), xytext=LEADER[arm], textcoords="data",
                    fontsize=8, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=2, shrinkB=4))
    else:
        ax.annotate(f"{lab} ({int(x)}, {int(y)})", (x, y), xytext=(8, -3),
                    textcoords="offset points", fontsize=8, color=INK)
rho = full.rank_mse.corr(full.rank_mae, method="spearman")
ax.set_xlabel("MSE rank (1 = best)")
ax.set_ylabel("MAE rank (1 = best)")
ax.set_xlim(0, n + 2)
ax.set_ylim(0, n + 2)
ax.grid(color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title(f"The ranking depends on the loss: 57 arms, Spearman ρ = {rho:.2f}", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, ncol=1)
fig.savefig(OUT / "rank_mse_vs_mae.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3. Aggregate test, per seed
runs = pd.read_csv(REC / "runs_aggregate-numeric-october.csv")
NAMES = {"lstm|numeric": "LSTM", "gru|numeric": "GRU", "mamba|numeric": "Mamba-2", "xgboost|numeric": "XGBoost",
         "s4_bidir|numeric": "S4 (bidir.)", "lightgbm|numeric": "LightGBM",
         "transformer_linhead|numeric": "Transformer", "s4|numeric": "S4", "gpt_linhead|numeric": "GPT",
         "seasonal_naive|none": "Seasonal Naive", "random_walk|none": "Random Walk"}
ma1 = runs[runs.arm == "moving_average|none"].re_mse.mean()
means = runs.groupby("arm").re_mse.mean()
order = [a for a in means.sort_values().index if a in NAMES]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
rng = np.random.default_rng(0)
for i, arm in enumerate(order):
    vals = runs[runs.arm == arm].re_mse.to_numpy()
    det = len(vals) == 1
    jitter = rng.uniform(-0.18, 0.18, len(vals)) if not det else np.zeros(1)
    ax.scatter(vals, i + jitter, s=16 if not det else 40, c=MUTED if det else BLUE,
               alpha=0.65 if not det else 1, edgecolors="none", zorder=3)
    ax.plot([vals.mean()] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.6, zorder=4)
ax.axvline(ma1, color=INK2, linewidth=1)
ax.text(ma1, len(order) - 0.35, f" Moving Average {ma1:.5f}", color=INK2, fontsize=8, va="bottom")
ax.set_yticks(range(len(order)), [NAMES[a] + (" (1 run)" if (runs.arm == a).sum() == 1 else "")
                                  for a in order], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("Test MSE (normalized), January–May 2025")
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
below = {NAMES[a]: int((runs[runs.arm == a].re_mse < ma1).sum()) for a in order}
below = {k: v for k, v in below.items() if v}
note = ", ".join(f"{v} {k} seeds" for k, v in below.items()) or "no seed"
ax.set_title(f"Aggregated series: Moving Average beats every model's mean ({note} fall below it)", loc="left")
fig.text(0.01, -0.02, "Dots = individual seeds; black tick = mean (Table 1). Gray = deterministic baselines.",
         fontsize=7.5, color=INK2)
fig.savefig(OUT / "aggregate_seed_spread.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 4. Window x lag sweep
sweep = pd.read_csv(REC / "fig2_sweep_cells.csv")
SW = [("lstm", "LSTM"), ("transformer", "Transformer"), ("s4", "S4"), ("s4_bidir", "S4 (bidir.)")]
W = [6, 12, 24, 36]
vmin, vmax = sweep.mse_x100.min(), sweep.mse_x100.max()
fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), gridspec_kw={"wspace": 0.12})
for ax, (m, lab) in zip(axes, SW):
    g = sweep[sweep.model == m].pivot(index="input_len", columns="lag_count", values="mse_x100").loc[W, W]
    ax.imshow(g.to_numpy(), cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="equal")
    bi, bj = np.unravel_index(np.argmin(g.to_numpy()), g.shape)
    for i in range(4):
        for j in range(4):
            v = g.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color=text_ink(SEQUENTIAL((v - vmin) / (vmax - vmin))),
                    fontweight="bold" if (i, j) == (bi, bj) else "normal")
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
    ax.set_xticks(range(4), W)
    ax.set_yticks(range(4), W if ax is axes[0] else [])
    ax.set_xlabel("lag count")
    if ax is axes[0]:
        ax.set_ylabel("input window (months)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    spread = (g.to_numpy().max() / g.to_numpy().min() - 1) * 100
    ax.set_title(f"{lab}  (spread {spread:.2f}%)", loc="left")
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(vmin, vmax)), ax=axes,
                  fraction=0.015, pad=0.015)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
cb.set_label("test MSE × 100 (lower is better)", color=INK2, fontsize=8)
fig.text(0.125, -0.04, "Embedding encoder, seeds 26/105/115; first supervised training target January 2015. "
         "Values rounded from unrounded seed means; outlined cell = best per model.", fontsize=7.5, color=INK2)
fig.savefig(OUT / "sweep_heatmaps.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
