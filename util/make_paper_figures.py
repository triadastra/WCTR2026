"""Composite paper figures from source-validated Exp 1 and Exp 2 runs.

Emits into graphs/_analysis/:
  exp2_leaderboard.png   - every model x variant ranked by Exp 2 norm-MSE, by family
  exp1_to_exp2_rank.png  - rank slopegraph: aggregated (Exp 1) vs multidim (Exp 2)
  accuracy_vs_params.png - efficiency frontier (neural models with param counts)
  family_distribution.png- Exp 2 norm-MSE distribution per model family

Run: python make_paper_figures.py
"""
import statistics as st
from pathlib import Path

import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# parents[1]: this script lives in util/, the repo root is one level up.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_paper_results import _exp1_rows, _exp2_rows, _aggregate_seeds
OUTDIR = ROOT / "graphs" / "_analysis"

FAMILY = {
    "mamba": "SSM", "mamba_nd": "SSM", "s4": "SSM", "s4nd": "SSM",
    "lstm": "RNN", "gru": "RNN",
    "transformer": "Attention", "gpt": "Attention",
    "xgboost": "Tree", "lightgbm": "Tree", "random_forest": "Tree",
    "moving_average": "Baseline", "random_walk": "Baseline", "seasonal_naive": "Baseline",
}
FAM_COLOR = {"SSM": "#34a853", "RNN": "#4285f4", "Attention": "#ea4335",
             "Tree": "#fbbc04", "Baseline": "#9aa0a6"}
FAM_ORDER = ["SSM", "RNN", "Attention", "Tree", "Baseline"]

def get_params(model, variant, raw):
    """Use the evaluated architecture count; historical constants are invalid."""
    if raw is None or not np.isfinite(float(raw)):
        return 0
    return int(raw)


def short_variant(v):
    return {"embeddings": "emb", "onehot": "1hot", "none": "",
            "cross_attention": "CA", "film_attention": "CA+FiLM",
            "cross_attention_2d": "2d", "cross_attention_3d": "3d",
            "cross_attention_4d": "4d", "2d_state": "2d", "2d": "2d(grp)",
            "matched_axial_2d": "2d matched", "matched_axial_3d": "3d matched",
            "matched_axial_4d": "4d matched",
            "3d": "3d", "4d": "4d"}.get(str(v), str(v))


def load_exp2():
    table = _aggregate_seeds(_exp2_rows(ROOT))
    return [dict(model=row.model, variant=row.variant, mse=row.norm_mse_combined,
                 std=row.norm_mse_std, params=getattr(row, "num_params", None))
            for row in table.itertuples()]


def load_exp1_mean():
    """Per-model Exp 1 driving loss, averaged over runs.

    This function used to return ``min(xs)`` -- the single best run of a model,
    pooled across both encoder variants. Exp 1 gives the neural arms 25 seeds
    each at a median CV of 37%, so a minimum is a best-of-25 draw on the TEST
    metric, and it was being placed on a rank chart beside deterministic
    single-run baselines that get exactly one draw. That is selection on the
    test set dressed as a ranking. The mean is what Exp1_Aggregated now reports
    and what the rank chart uses.

    (The old expression was also dead code -- a conditional whose guard was the
    literal False, so only the minimum branch could ever run.)
    """
    rows, _ = _exp1_rows(ROOT)
    agg = {}
    for row in rows:
        agg.setdefault(row["model"], []).append(float(row["norm_mse_combined"]))
    return {m: st.mean(xs) for m, xs in agg.items()}


def fig_leaderboard(e2):
    # Dot plot (point = mean, whiskers = std), NOT bars: values are in a narrow
    # band so a bar chart would need a truncated x-axis where bar length is
    # meaningless (best model = shortest bar).
    data = sorted(e2, key=lambda d: d["mse"])
    labels = [f"{d['model']} {short_variant(d['variant'])}".strip() for d in data]
    vals = [d["mse"] for d in data]
    errs = [d["std"] for d in data]
    fig, ax = plt.subplots(figsize=(10, 13))
    n = len(data)
    y = np.arange(n)[::-1]
    best = vals[0]
    ax.axvline(best, ls="--", lw=1, color="#202124", zorder=1)
    for yi, d, v, e in zip(y, data, vals, errs):
        c = FAM_COLOR[FAMILY.get(d["model"], "Baseline")]
        ax.errorbar(v, yi, xerr=e, fmt="o", ms=6.5, color=c, ecolor=c,
                    elinewidth=1.0, capsize=3, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("Exp 2 per-combo normalized MSE  (point = 3-seed mean, whiskers = std; lower = better)")
    ax.set_title("Multidimensional benchmark leaderboard (Exp 2)\ndashed line = best")
    ax.set_xlim(0.0485, max(v + e for v, e in zip(vals, errs)) + 0.003)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=FAM_COLOR[f], label=f)
               for f in FAM_ORDER]
    ax.legend(handles=handles, title="family", loc="lower right", frameon=False)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(OUTDIR / "exp2_leaderboard.png", dpi=160)
    print("wrote exp2_leaderboard.png")


def fig_rank_shift(e2, e1):
    # one number per model: best variant in each experiment
    e2best = {}
    for d in e2:
        e2best[d["model"]] = min(e2best.get(d["model"], 9), d["mse"])
    models = [m for m in e1 if m in e2best]
    r1 = {m: i for i, m in enumerate(sorted(models, key=lambda m: e1[m]))}
    r2 = {m: i for i, m in enumerate(sorted(models, key=lambda m: e2best[m]))}
    n = len(models)
    fig, ax = plt.subplots(figsize=(9, 8))
    for m in models:
        c = FAM_COLOR[FAMILY.get(m, "Baseline")]
        ax.plot([0, 1], [n - r1[m], n - r2[m]], "-o", color=c, lw=2, ms=6)
        ax.text(-0.02, n - r1[m], f"{m} ({e1[m]:.4f})", ha="right", va="center", fontsize=8)
        ax.text(1.02, n - r2[m], f"{m} ({e2best[m]:.4f})", ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(0.3, n + 0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Exp 1\n(aggregated single series)",
                                               "Exp 2\n(multidimensional)"])
    ax.set_yticks([]); ax.set_title("Descriptive rank comparison across tasks\n"
                                    "Exp 1: mean across variants; Exp 2: lowest observed variant")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([0], [0], color=FAM_COLOR[f], lw=3) for f in FAM_ORDER]
    ax.legend(handles, FAM_ORDER, title="family", loc="upper center",
              ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(OUTDIR / "exp1_to_exp2_rank.png", dpi=160)
    print("wrote exp1_to_exp2_rank.png")


def fig_acc_params(e2):
    # EVERY trained run (model x variant). Trees + baselines have ~0 gradient
    # params and sit at x=0 (listed in a box); parametric runs (RNN/Attn/SSM)
    # at their real counts, with S4-family counts resolved from checkpoints.
    fig, ax = plt.subplots(figsize=(12, 7.5))
    drop = {"random_walk", "seasonal_naive"}  # weak outlier baselines, excluded
    zero, param = [], []
    for d in e2:
        if d["model"] in drop:
            continue
        p = get_params(d["model"], d["variant"], d["params"])
        (zero if p == 0 else param).append((d["model"], dict(d, params=p)))

    def ci(d):  # 95% CI half-width from the 3-seed std (Student-t, 2 dof)
        return 4.302653 * d.get("std", 0.0) / (3 ** 0.5)

    # parametric runs: dots with vertical 95% CI whiskers + compact labels
    for m, d in param:
        c = FAM_COLOR[FAMILY.get(m, "Baseline")]
        ax.errorbar(d["params"] / 1e6, d["mse"], yerr=ci(d), fmt="o", ms=7,
                    color=c, ecolor=c, elinewidth=0.9, capsize=2, alpha=0.85,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    # zero-param runs: dots at x=0, enumerated in the box
    for m, d in zero:
        c = FAM_COLOR[FAMILY.get(m, "Baseline")]
        ax.errorbar(0, d["mse"], yerr=ci(d), fmt="o", ms=7, color=c, ecolor=c,
                    elinewidth=0.9, capsize=2, alpha=0.9, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3, clip_on=False)
    if zero:
        lines = ["non-parametric (0 params):"] + [
            f"  {m} {short_variant(d['variant'])}".rstrip() + f"  {d['mse']:.4f}"
            for m, d in sorted(zero, key=lambda kv: kv[1]["mse"])]
        ax.text(0.30, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
                va="top", ha="left", color="#3c4043",
                bbox=dict(boxstyle="round", fc="white", ec="#9aa0a6", alpha=0.9))
    # Pareto frontier over ALL parametric runs
    front, best_so = [], 9
    for m, d in sorted(param, key=lambda kv: kv[1]["params"]):
        if d["mse"] < best_so:
            front.append((d["params"] / 1e6, d["mse"])); best_so = d["mse"]
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.plot(fx, fy, ls="--", color="#202124", lw=1, zorder=2, label="Pareto frontier")
    ax.set_xlabel("trainable parameters (millions)  —  0 = non-parametric (trees, baselines)")
    ax.set_ylabel("Exp 2 per-combo normalized MSE  (lower = better)")
    ax.set_title("Accuracy vs. model size — every trained run (all model×variant)\n"
                 "dots = 3-seed mean, whiskers = 95% CI")
    ax.set_xlim(-0.04, max(d["params"] for _, d in param) / 1e6 * 1.1)
    # zoom to where the models actually live; the two weak baselines
    # (random_walk 0.066, seasonal_naive 0.084) are in the box, off-axis.
    ax.set_ylim(0.0485, 0.0545)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=FAM_COLOR[f], label=f)
               for f in FAM_ORDER]
    handles.append(plt.Line2D([0], [0], ls="--", color="#202124", label="Pareto frontier"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(OUTDIR / "accuracy_vs_params.png", dpi=160)
    print("wrote accuracy_vs_params.png")


def fig_params_table(e2):
    """Full parameter-count table for every trained model x variant. Writes
    params_table.csv and a rendered params_table.png."""
    rows = []
    for d in sorted(e2, key=lambda d: (FAM_ORDER.index(FAMILY.get(d["model"], "Baseline")),
                                       d["model"], d["variant"])):
        fam = FAMILY.get(d["model"], "Baseline")
        p = get_params(d["model"], d["variant"], d["params"])
        if fam == "Tree":
            disp = "— (tree ensemble)"
        elif fam == "Baseline":
            disp = "0 (deterministic)"
        else:
            disp = f"{p:,}" if p else "unavailable"
        rows.append((fam, d["model"], short_variant(d["variant"]) or "—", disp, p))
    # CSV
    import csv as _csv
    with open(OUTDIR / "params_table.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["family", "model", "variant", "params"])
        for r in rows:
            w.writerow(r[:4])
    # PNG table
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(rows) + 1))
    ax.axis("off")
    cells = [[r[1], r[2], r[3]] for r in rows]
    tbl = ax.table(cellText=cells, colLabels=["model", "variant", "params"],
                   cellLoc="left", colWidths=[0.34, 0.30, 0.36], loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.25)
    for j in range(3):
        tbl[0, j].set_facecolor("#202124"); tbl[0, j].set_text_props(color="white", weight="bold")
    for i, r in enumerate(rows, start=1):
        col = FAM_COLOR[r[0]]
        for j in range(3):
            tbl[i, j].set_facecolor(col); tbl[i, j].set_alpha(0.18)
    ax.set_title("Trained-model parameter counts (all variants)\n"
                 "S4 family from checkpoints; trees non-parametric; baselines deterministic",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(OUTDIR / "params_table.png", dpi=160, bbox_inches="tight")
    print(f"wrote params_table.csv + params_table.png ({len(rows)} rows)")


def fig_family(e2):
    byfam = {f: [] for f in FAM_ORDER}
    for d in e2:
        byfam[FAMILY.get(d["model"], "Baseline")].append(d["mse"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, f in enumerate(FAM_ORDER):
        ys = byfam[f]
        if not ys:
            continue
        xs = np.random.default_rng(0).normal(i, 0.06, len(ys))
        ax.scatter(xs, ys, color=FAM_COLOR[f], s=45, edgecolor="white", zorder=3, alpha=0.9)
        ax.hlines(st.mean(ys), i - 0.25, i + 0.25, color="#202124", lw=2, zorder=4)
    ax.set_xticks(range(len(FAM_ORDER))); ax.set_xticklabels(FAM_ORDER)
    ax.set_ylabel("Exp 2 per-combo normalized MSE  (lower = better)")
    ax.set_title("Per-family performance distribution (Exp 2)\n"
                 "black bar = mean across the declared arms in each family")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(OUTDIR / "family_distribution.png", dpi=160)
    print("wrote family_distribution.png")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    e2 = load_exp2()
    e1 = load_exp1_mean()
    fig_leaderboard(e2)
    fig_rank_shift(e2, e1)
    fig_acc_params(e2)
    fig_family(e2)
    fig_params_table(e2)


if __name__ == "__main__":
    main()
