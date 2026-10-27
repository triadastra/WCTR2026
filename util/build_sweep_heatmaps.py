#!/usr/bin/env python3
"""Exp 3 sweep heatmaps: one 4x4 (window x lag) matplotlib heatmap per model,
colored by three-seed mean normalized test MSE. Reads the per-cell
metrics.json under outputs/sweep/. Run from the repo root on the VM.

Every cell must contain the same pinned seeds; incomplete coverage is rejected
rather than silently mixing seed variance into selected cells."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.provenance import assert_corrected_runs  # noqa: E402
RUNS = Path("outputs/sweep")
# graphs/exp3_sweep/ is where the repo keeps these and where graphs/README.md
# points; writing to graphs/ meant every regeneration had to be moved by hand,
# and a stale copy in the real location outlived the run that replaced it.
GRAPHS = Path("graphs") / "exp3_sweep"; GRAPHS.mkdir(parents=True, exist_ok=True)
WINDOWS = [6, 12, 24, 36]
LAGS = [6, 12, 24, 36]
MODELS = ["lstm", "transformer", "s4", "s4_bidir"]
VARIANT = "embeddings"
# Must match build_paper_results.SWEEP_RE and both canonical launchers.
SEEDS = [26, 105, 115]


def load_cells():
    # Reuse the publication gate: exact grid, identity, source, optimizer,
    # checkpoint selection, prediction files, and all three seeds.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.build_paper_results import _sweep_rows
    rows = _sweep_rows(RUNS)
    if not rows:
        raise RuntimeError("Exp 3 has no validated complete sweep")
    return [(r['model'], r['variant'], r['window'], r['lag_count'],
             r['seed'], r['norm_mse_combined']) for r in rows]


def main():
    # Provenance gate: never publish the Exp 3 sweep heatmaps from artifacts
    # evaluated on the pre-correction, survivorship-biased cohort. Only when the
    # sweep has actually been run -- an absent outputs/sweep/ is "nothing to do
    # yet", which load_cells() already reports, not a provenance failure.
    if RUNS.is_dir():
        assert_corrected_runs(RUNS, what="the Exp 3 sweep heatmaps")
    rows = load_cells()
    for model in MODELS:
        M = np.full((len(WINDOWS), len(LAGS)), np.nan)
        N = np.zeros((len(WINDOWS), len(LAGS)), dtype=int)
        for i, w in enumerate(WINDOWS):
            for j, l in enumerate(LAGS):
                vals = [r[5] for r in rows if r[0] == model and r[1] == VARIANT
                        and r[2] == w and r[3] == l and r[4] in SEEDS]
                if vals:
                    M[i, j] = float(np.mean(vals)); N[i, j] = len(vals)
        done = int(np.sum(~np.isnan(M)))
        if done == 0:
            print(f"[skip] {model}: no cells yet")
            continue
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        im = ax.imshow(M, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(LAGS)), LAGS)
        ax.set_yticks(range(len(WINDOWS)), WINDOWS)
        ax.set_xlabel("lag_count (# lag features)")
        ax.set_ylabel("input window (months)")
        counts = sorted({int(N[i, j]) for i in range(len(WINDOWS))
                         for j in range(len(LAGS)) if not np.isnan(M[i, j])})
        if done != 16 or counts != [len(SEEDS)]:
            raise RuntimeError(
                f"{model}: incomplete sweep ({done}/16 cells; seed counts {counts}); "
                "refusing a ragged heatmap"
            )
        label = f"mean of {len(SEEDS)} seeds"
        ax.set_title(f"{model} — Exp 3 sweep: normalized test MSE\n({label}, {VARIANT}; lower = better)")
        for i in range(len(WINDOWS)):
            for j in range(len(LAGS)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.4f}\n(n={N[i, j]})", ha="center", va="center",
                            color="white", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized test MSE")
        fig.tight_layout()
        out = GRAPHS / f"sweep_heatmap_{model}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"[ok] {out}  ({done}/16 cells filled)")


if __name__ == "__main__":
    main()
