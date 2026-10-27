"""Recompute graphs/_analysis/ from the complete validated Exp 2 artifacts.

Cached CSVs and existing per-run PNGs are not publication evidence. The legacy
collect_curated helper is retained for manual browsing only; the canonical
entry point never copies those potentially stale images into a new report.

ROOT is derived from this file's location (repo root = parents[1] of
scripts/), so the script is portable and can be run from anywhere.
"""
import sys
import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]   # repo root (scripts/ -> ..)
sys.path.insert(0, str(ROOT))
from scripts.build_paper_results import _exp2_rows, _aggregate_seeds  # noqa: E402
RUNS = ROOT / "outputs/full"
GRAPHS = ROOT / "graphs"
# Fixed-name plots curated for every run (copied verbatim if present).
CURATED = [
    "aggregate_value.png",
    "aggregate_weight.png",
    "loss_curve.png",
    "residual_value.png",        # optional: residual over test months
    "residual_weight.png",       # optional
    "scatter_value.png",         # optional: predicted-vs-actual scatter
    "scatter_weight.png",        # optional
]
# Per-series plots produced by inference are named series_<i>_<key>.png.
SERIES_RE = re.compile(r"^series_\d+_.*\.png$")
# Strip any seed suffix (_seed42, _seed7, ...), not just _seed42.
SEED_RE = re.compile(r"_seed\d+$")


def clean_name(run_dir_name: str) -> str:
    n = SEED_RE.sub("", run_dir_name)           # any _seed<digits> suffix
    n = n.replace("_none", "")                  # baselines: moving_average_none -> moving_average
    if n.startswith("s4_s4nd_"):
        n = "s4nd_" + n[len("s4_s4nd_"):]       # s4_s4nd_2d -> s4nd_2d
    elif n.startswith("s4_s4_"):
        n = "s4_" + n[len("s4_s4_"):]           # s4_s4_embeddings -> s4_embeddings
    return n


def collect_curated():
    copied = 0
    series = 0
    if not RUNS.exists():
        print(f"[curated] runs dir not found: {RUNS} (skipping)")
        return
    # One canonical seed per arm. clean_name strips the seed, so iterating every
    # run dir made the three seeds overwrite each other file-by-file in
    # lexicographic order (seed105, seed115, seed26) — a folder could silently
    # mix plots from different seeds. Pick the lowest seed present per arm
    # instead (build_exp2_tree.py handles the exp2 tree with explicit labels).
    by_arm = {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or not SEED_RE.search(d.name):
            continue
        seed = int(SEED_RE.search(d.name).group(0)[len("_seed"):])
        arm = clean_name(d.name)
        if arm not in by_arm or seed < by_arm[arm][0]:
            by_arm[arm] = (seed, d)
    for arm, (seed, d) in sorted(by_arm.items()):
        plots = d / "plots"
        if not plots.is_dir():
            continue
        dest = GRAPHS / arm
        dest.mkdir(parents=True, exist_ok=True)
        # Fixed-name curated plots (guarded: only copy what exists).
        for fn in CURATED:
            src = plots / fn
            if src.exists():
                shutil.copy(src, dest / fn)
                copied += 1
        # Per-series plots series_<i>_<key>.png (optional / guarded).
        for src in sorted(plots.glob("series_*.png")):
            if SERIES_RE.match(src.name):
                shutil.copy(src, dest / src.name)
                series += 1
    print(f"[curated] copied {copied} fixed-name + {series} series plot files into graphs/")


def load_results():
    """Use the workbook's exact validated population and seed-mean arithmetic."""
    table = _aggregate_seeds(_exp2_rows(ROOT))
    return {(r["model"], r["variant"]): r for r in table.to_dict("records")}


def encoder_ratio_heatmap(d):
    models = ["lstm", "gru", "transformer", "gpt", "dss", "s4", "s4_bidir", "xgboost", "lightgbm"]
    metrics = [("mae", "MAE"), ("norm_mse_combined", "MSE")]
    M = np.full((len(models), len(metrics)), np.nan)
    for i, m in enumerate(models):
        oh, em = d.get((m, "onehot")), d.get((m, "embeddings"))
        if not oh or not em:
            continue
        for j, (k, _) in enumerate(metrics):
            try:
                M[i, j] = float(oh[k]) / float(em[k])
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    fig, ax = plt.subplots(figsize=(4.2, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0.85, vmax=1.15, aspect="auto")
    ax.set_xticks(range(len(metrics)), [n for _, n in metrics])
    ax.set_yticks(range(len(models)), models)
    ax.set_title("Encoder ratio  (one-hot / embeddings)\n~1.0 = similar observed mean loss", fontsize=10)
    for i in range(len(models)):
        for j in range(len(metrics)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("one-hot / embeddings")
    fig.tight_layout()
    out = GRAPHS / "_analysis" / "encoder_ratio_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analysis] wrote {out.name}")


def leaderboard(d):
    items = []
    for (model, variant), r in d.items():
        v = r.get("norm_mse_combined")
        if v not in (None, ""):
            label = f"{model}/{variant}" if variant not in ("none",) else model
            items.append((label, float(v)))
    items.sort(key=lambda x: x[1])
    labels = [a for a, _ in items]
    vals = [b for _, b in items]
    colors = ["#2b8cbe" if "s4" in label else "#bdbdbd" for label in labels]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("normalized test MSE (lower = better)")
    ax.set_title("Benchmark leaderboard (MinMax-normalized, n=4590, seed-mean)\nblue = S4 family")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)
    fig.tight_layout()
    out = GRAPHS / "_analysis" / "leaderboard.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[analysis] wrote {out.name}")


if __name__ == "__main__":
    d = load_results()  # validate before publishing any graphs
    encoder_ratio_heatmap(d)
    leaderboard(d)
    print("done. graphs/ tree:")
    if GRAPHS.exists():
        for p in sorted(GRAPHS.iterdir()):
            n = len(list(p.glob("*.png"))) if p.is_dir() else 0
            print(f"  {p.name}/  ({n} png)")
