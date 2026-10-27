#!/usr/bin/env python
"""Lane-weighted comparisons against the fixed moving-average reference.

Inputs pass the complete source/completion gate. The primary mean averages
valid series-time observations, with state-clustered uncertainty. Equal-state
differences are labeled sensitivity results. Holm and exploratory BH adjust
the complete comparison family using wild-cluster p-values. With eight states,
shared national shocks and prior test inspection still limit interpretation.
--pair renders MSE and MAE as separate exploratory families.
"""
import math
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

from ext.eval.resampling import (  # noqa: E402
    holm, benjamini_hochberg)
from make_significance import load_model_losses, paired_comparison         # noqa: E402

OUTDIR = ROOT / "graphs" / "_analysis"
BASELINE = "moving_average_none"
BASELINES = ("moving_average_none", "random_walk_none", "seasonal_naive_none")

FAMILY = {
    "lstm": "RNN", "gru": "RNN",
    "transformer": "Attention", "gpt": "Attention",
    "mamba": "SSM", "s4": "SSM", "s4_bidir": "SSM", "mamba_nd": "SSM", "s4nd": "SSM",
    "xgboost": "Tree", "lightgbm": "Tree", "random_forest": "Tree",
}
FAM_ORDER = ["SSM", "Attention", "RNN", "Tree"]

# Both naming schemes onto one axis -- see THE HEATMAP in the module docstring.
STRUCT = {
    "onehot": "flat\n(one-hot)", "embeddings": "flat\n(embed)",
    "cross_attention_2d": "2D", "cross_attention_3d": "3D", "cross_attention_4d": "4D",
    "2d": "2D", "2d_state": "2D-state", "3d": "3D", "4d": "4D",
}
STRUCT_ORDER = ["flat\n(one-hot)", "flat\n(embed)", "2D", "2D-state", "3D", "4D"]


def split_arm(arm):
    """'s4_cross_attention_2d' -> ('s4', 'cross_attention_2d'). Longest model wins
    so mamba_nd is not parsed as mamba."""
    for m in sorted(FAMILY, key=len, reverse=True):
        if arm == m or arm.startswith(m + "_"):
            return m, arm[len(m) + 1:]
    if arm.startswith("s4nd_"):
        return "s4nd", arm[5:]
    return arm, ""


def _grid_from_rows(rows):
    """rows: list of dicts with model/variant/improvement_pct/sig flags."""
    import pandas as pd
    t = pd.DataFrame(rows)
    t["struct"] = t.variant.map(STRUCT)
    t["family"] = t.model.map(FAMILY)
    grid = t.dropna(subset=["struct"]).copy()
    grid["fam_rank"] = grid.family.map({f: i for i, f in enumerate(FAM_ORDER)})
    models = (grid.groupby(["fam_rank", "family", "model"]).mean_metric.min()
              .reset_index().sort_values(["fam_rank", "mean_metric"]))
    row_index = list(models.model)
    M = np.full((len(row_index), len(STRUCT_ORDER)), np.nan)
    RAW = np.zeros_like(M, dtype=bool)
    FDR = np.zeros_like(M, dtype=bool)
    for _, r in grid.iterrows():
        i, j = row_index.index(r.model), STRUCT_ORDER.index(r.struct)
        if not np.isnan(M[i, j]):
            raise RuntimeError("Multiple arms map to one heatmap cell")
        if np.isnan(M[i, j]):
            M[i, j] = r.improvement_pct
            RAW[i, j] = bool(r.sig_raw)
            FDR[i, j] = bool(r.get("sig_fdr", False))
    return row_index, dict(zip(models.model, models.family)), M, RAW, FDR


def pair_mode():
    """MSE and MAE: descriptive pooled improvement and state-clustered tests."""
    from matplotlib.colors import TwoSlopeNorm
    import make_significance as ms

    panels = []
    for metric, label in (("mse", "MSE (driving loss)"), ("mae", "MAE")):
        data = ms.load_model_losses(metric=metric)
        means = {mv: float(v[0].mean()) for mv, v in data.items()}
        base, base_mean = data[BASELINE][0], means[BASELINE]
        arms = [a for a in sorted(means, key=means.get) if a not in BASELINES]
        rows, raw_p = [], []
        for a in arms:
            comparison = paired_comparison(base, data[a][0])
            model, variant = split_arm(a)
            p = comparison["p_wild_cluster"]
            rows.append({"model": model, "variant": variant,
                         "mean_metric": means[a],
                         "improvement_pct": (base_mean - means[a]) / base_mean * 100,
                         "beats": comparison["mean_lane_delta"] > 0, "p": p})
            raw_p.append(p if np.isfinite(p) else 1.)
        q = benjamini_hochberg(raw_p)
        for r, qq in zip(rows, q):
            r["sig_raw"] = bool(r["beats"] and r["p"] < 0.05)
            r["sig_fdr"] = bool(r["beats"] and qq < 0.05)
            r["q"] = qq
        best_q = float(min(qq for qq in q))
        clear = [(r["model"], r["variant"], r["improvement_pct"], r["q"])
                 for r in rows if r["sig_fdr"]]
        print(f"[pair] {label}: {len(clear)} arms clear FDR q<0.05")
        for m, v, imp, qq in sorted(clear, key=lambda x: x[3]):
            print(f"        {m} {v}  {imp:+.2f}%  q={qq:.4f}")
        panels.append((label, rows, best_q, len(arms)))

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 0.62 * 11 + 3.0))
    cmap = plt.get_cmap("PiYG").copy()
    cmap.set_bad("#e9e9e9")
    # Row order comes from the MSE panel and is reused for MAE so the rows
    # align across panels.
    row_index, fam_of, M0, RAW0, FDR0 = _grid_from_rows(panels[0][1])
    grids = [(M0, RAW0, FDR0)]
    ri1, _, M1raw, RAW1raw, FDR1raw = _grid_from_rows(panels[1][1])
    idx = [ri1.index(m) for m in row_index]
    grids.append((M1raw[idx], RAW1raw[idx], FDR1raw[idx]))

    n_rows, n_cols = len(row_index), len(STRUCT_ORDER)
    for k, (ax, (label, rows, best_q, n_arms), (M, RAW, FDR)) in enumerate(
            zip(axes, panels, grids)):
        vmax = max(float(np.nanmax(M)) * 1.05, 0.5)
        vmin = min(float(np.nanmin(M)) * 1.15, -0.5 * vmax)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(n_cols), STRUCT_DISPLAY, fontsize=8.5)
        if k == 0:
            ax.set_yticks(range(n_rows),
                          [DISPLAY.get(m, m) for m in row_index], fontsize=11)
        else:
            ax.set_yticks(range(n_rows), [""] * n_rows)
            ax.tick_params(axis="y", length=0)
        n_raw = int(RAW.sum())
        n_fdr = int(FDR.sum())
        note = (f"✱ = FDR q<0.05 ({n_fdr} cells)" if n_fdr else
                f"no FDR-significant improvement (min two-sided q = {best_q:.2g})")
        ax.set_title(f"{label}\n† = uncorrected p<0.05 ({n_raw} cells); "
                     + note, fontsize=10.5, color="#333333", pad=10)
        for i in range(n_rows):
            for j in range(n_cols):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "—", ha="center", va="center",
                            color="#9a9a9a", fontsize=10)
                    continue
                r_, g_, b_, _ = cmap(norm(M[i, j]))
                lum = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_
                mark = "✱" if FDR[i, j] else ("†" if RAW[i, j] else "")
                ax.text(j, i, f"{M[i, j]:+.2f}{mark}", ha="center",
                        va="center", fontsize=9.5,
                        color="white" if lum < 0.45 else "#111111",
                        fontweight="bold" if (RAW[i, j] or FDR[i, j])
                        else "normal")
        for i in range(1, n_rows):
            if fam_of[row_index[i]] != fam_of[row_index[i - 1]]:
                ax.axhline(i - 0.5, color="#222222", lw=2.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.axhline(0.0, color="#222222", lw=1.0)

    fig.suptitle("Comparison against moving average (state-clustered) — "
                 "MSE and MAE", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.045,
             "Improvement over the moving average (%), per panel's own loss; "
             "color scales are per-panel. Statistic per panel: wild bootstrap on "
             "state-clustered differentials, exploratory BH-FDR.",
             ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.022,
             "Colors and marks: lane-weighted improvements. Non-significance is not equivalence.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0.05, 0.06, 1.0, 0.94))
    seen = {}
    for i, m in enumerate(row_index):
        seen.setdefault(fam_of[m], []).append(i)
    inv = fig.transFigure.inverted()
    for fam, idx_ in seen.items():
        _, y = inv.transform(axes[0].transData.transform(
            (0, float(np.mean(idx_)))))
        fig.text(0.022, y, fam, ha="center", va="center", fontsize=10,
                 fontweight="bold", rotation=90, color="#222222")
    out = OUTDIR / "vs_baseline_pair.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[vs-baseline] wrote {out.name}")
    return 0


def baseline_table(data):
    base = data[BASELINE][0]
    means = {mv: float(v[0].mean()) for mv, v in data.items()}
    base_mean = means[BASELINE]
    arms = [a for a in sorted(means, key=means.get) if a not in BASELINES]
    raw_p, rows = [], []
    for a in arms:
        comparison = paired_comparison(base, data[a][0])
        model, variant = split_arm(a)
        rows.append({**comparison,
            "arm": a, "model": model, "variant": variant,
            "family": FAMILY.get(model, "?"),
            "mean_norm_mse": means[a],
            "improvement_pct": (base_mean - means[a]) / base_mean * 100,
            "beats_baseline": bool(comparison["mean_lane_delta"] > 0),
            "lower_pooled_loss": bool(means[a] < base_mean),
        })
        p = comparison["p_wild_cluster"]
        raw_p.append(p if np.isfinite(p) else 1.)

    # Report both family-wise and exploratory FDR error control.
    for r, ph, q in zip(rows, holm(raw_p), benjamini_hochberg(raw_p)):
        r["p_holm"] = ph
        r["q_bh"] = q
        # Significant AND in the winning direction. A significant loss to a
        # moving average is not a win, and collapsing the two would read as one.
        r["verdict"] = ("degenerate" if r["degenerate"] else
                        "beats MA" if r["beats_baseline"] and q < 0.05 else
                        "LOSES to MA" if (not r["beats_baseline"]) and q < 0.05 else
                        "n.s.")
        r["holm_verdict"] = ("beats MA" if r["beats_baseline"] and ph < 0.05 else
                             "LOSES to MA" if (not r["beats_baseline"]) and ph < 0.05 else
                             "n.s.")

    tbl = pd.DataFrame(rows).sort_values("mean_norm_mse").reset_index(drop=True)
    return tbl


def main():
    if "--plot-only" in sys.argv:
        print("--plot-only is unsupported; regenerate from validated predictions.")
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if "--pair" in sys.argv:
        return pair_mode()
    data = load_model_losses()
    if not data:
        print("[vs-baseline] no predictions under outputs/full/")
        return 1
    if BASELINE not in data:
        print(f"[vs-baseline] baseline {BASELINE} not found; have "
              f"{[k for k in data if 'moving' in k or 'naive' in k or 'walk' in k]}")
        return 1

    tbl = baseline_table(data)
    base_mean = float(data[BASELINE][0].mean())
    tbl.to_csv(OUTDIR / "vs_baseline.csv", index=False)

    n_beat = (tbl.verdict == "beats MA").sum()
    n_lose = (tbl.verdict == "LOSES to MA").sum()
    n_ns = (tbl.verdict == "n.s.").sum()
    print(f"[vs-baseline] reference {BASELINE} (norm_mse {base_mean:.5f})")
    n_holm = (tbl.holm_verdict == "beats MA").sum()
    print(f"[vs-baseline] {len(tbl)} arms, exploratory BH-FDR: {n_beat} beat MA, {n_lose} lose, {n_ns} n.s.")
    print(f"[vs-baseline] {n_holm} arms clear Holm family-wise correction")
    print("\n" + tbl[["arm", "improvement_pct", "p_cluster_t", "q_bh", "p_holm", "verdict"]]
          .head(10).to_string(index=False))
    print("\n[vs-baseline] not beating MA:")
    print(tbl[tbl.verdict != "beats MA"][["arm", "improvement_pct", "p_wild_cluster", "q_bh", "verdict"]]
          .to_string(index=False))

    make_heatmap(tbl, base_mean)
    return 0


# Publication display names; internal keys stay snake_case everywhere else.
DISPLAY = {
    "lstm": "LSTM", "gru": "GRU", "transformer": "Transformer", "gpt": "GPT",
    "mamba": "Mamba", "s4": "S4", "s4_bidir": "S4-Bidirectional", "mamba_nd": "Mamba-ND", "s4nd": "S4ND",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "Random Forest",
}
STRUCT_DISPLAY = ["Flat\n(one-hot)", "Flat\n(embeddings)", "2D", "2D-state", "3D", "4D"]


def make_heatmap(tbl, base_mean):
    from matplotlib.colors import TwoSlopeNorm

    t = tbl.copy()
    t["struct"] = t.variant.map(STRUCT)
    grid = t.dropna(subset=["struct"]).copy()

    # Row order: family block, then best arm first within each block.
    grid["fam_rank"] = grid.family.map({f: i for i, f in enumerate(FAM_ORDER)})
    models = (grid.groupby(["fam_rank", "family", "model"]).mean_norm_mse.min()
              .reset_index().sort_values(["fam_rank", "mean_norm_mse"]))
    row_index = list(models.model)

    M = np.full((len(row_index), len(STRUCT_ORDER)), np.nan)
    SIG = np.zeros_like(M, dtype=bool)   # clears FDR q < 0.05
    RAW = np.zeros_like(M, dtype=bool)   # clears uncorrected p < 0.05
    n_arms = len(tbl)
    best_q = float(tbl.q_bh.min())
    for _, r in grid.iterrows():
        i, j = row_index.index(r.model), STRUCT_ORDER.index(r.struct)
        if not np.isnan(M[i, j]):
            raise RuntimeError("Multiple arms map to one heatmap cell")
        if np.isnan(M[i, j]):
            M[i, j] = r.improvement_pct
            SIG[i, j] = (r.verdict == "beats MA")
            RAW[i, j] = bool(r.beats_baseline and r.p_wild_cluster < 0.05)

    n_rows, n_cols = len(row_index), len(STRUCT_ORDER)
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * n_rows + 3.0))

    # Asymmetric diverging norm centred at 0. The data run -0.6..+5.6; a
    # symmetric +/-5.6 scale rendered every losing tree arm near-white. PiYG is
    # ColorBrewer diverging (CVD-safe, unlike RdYlGn): green = beats MA,
    # magenta = loses to MA, white = the baseline itself.
    vmax = max(float(np.nanmax(M)) * 1.05, 0.5)
    vmin = min(float(np.nanmin(M)) * 1.6, -0.5 * vmax)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("PiYG").copy()
    cmap.set_bad("#e9e9e9")
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

    # Crisp white cell borders.
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(n_cols), STRUCT_DISPLAY, fontsize=11)
    ax.set_yticks(range(n_rows), [DISPLAY.get(m, m) for m in row_index],
                  fontsize=11)
    ax.set_xlabel("Structural dimension of the combination encoder", fontsize=11)

    n_fdr = int(SIG.sum())
    n_raw = int(RAW.sum())
    # State what the marks mean AND what the evidence actually is. Claiming
    # "✱ = significant" while nothing qualifies invites the reader to assume the
    # unmarked cells failed a weaker bar than they did.
    sig_note = (f"✱ = FDR q<0.05 ({n_fdr} cells) · " if n_fdr else
                f"no FDR-significant improvement across {n_arms} arms "
                f"(min two-sided q = {best_q:.2g}) · ")
    fig.suptitle("State-blocked comparison against moving average",
                 fontsize=15, fontweight="bold", y=0.975)
    ax.set_title("Improvement in normalized MSE over the moving-average "
                 "baseline (%)\n"
                 + sig_note + f"† = uncorrected p<0.05 ({n_raw} cells)",
                 fontsize=10.5, color="#333333", pad=12)

    # Cell text: colour by the cell's own luminance so values stay readable on
    # both the darkest greens and the pale mid-range.
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        color="#9a9a9a", fontsize=11)
                continue
            r_, g_, b_, _ = cmap(norm(M[i, j]))
            lum = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_
            mark = "✱" if SIG[i, j] else ("†" if RAW[i, j] else "")
            ax.text(j, i, f"{M[i, j]:+.2f}{mark}", ha="center", va="center",
                    fontsize=10.5, color="white" if lum < 0.45 else "#111111",
                    fontweight="bold" if (SIG[i, j] or RAW[i, j]) else "normal")

    # Family separators. Labels go in FIGURE coordinates, left of the tick
    # labels -- placing them in data coords put "Attention" on top of
    # "transformer".
    fam_of = dict(zip(models.model, models.family))
    for i in range(1, n_rows):
        if fam_of[row_index[i]] != fam_of[row_index[i - 1]]:
            ax.axhline(i - 0.5, color="#222222", lw=2.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("% better than moving average", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9.5)
    # Colorbar axes are in DATA units (vmin..vmax), not normalized [0, 1].
    cbar.ax.axhline(0.0, color="#222222", lw=1.0)

    fig.tight_layout(rect=(0.06, 0.06, 1.0, 0.95))
    fig.text(0.5, 0.032,
             "Statistic: wild bootstrap on lane-weighted MSE differentials "
             "(eight state clusters, exploratory); BH-FDR across all arms.",
             ha="center", fontsize=8.5, color="#555555")
    fig.text(0.5, 0.013, "Colors: pooled loss; marks: lane-weighted loss. Non-significance is not equivalence.",
             ha="center", fontsize=8.5, color="#555555")

    seen = {}
    for i, m in enumerate(row_index):
        seen.setdefault(fam_of[m], []).append(i)
    inv = fig.transFigure.inverted()
    for fam, idx in seen.items():
        _, y = inv.transform(ax.transData.transform((0, float(np.mean(idx)))))
        fig.text(0.03, y, fam, ha="center", va="center",
                 fontsize=11, fontweight="bold", rotation=90, color="#222222")
    out = OUTDIR / "vs_baseline_heatmap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    filled = int(np.sum(~np.isnan(M)))
    print(f"\n[vs-baseline] wrote {out.name} "
          f"({filled}/{M.size} cells = {filled / M.size * 100:.0f}% filled)")


if __name__ == "__main__":
    raise SystemExit(main())
