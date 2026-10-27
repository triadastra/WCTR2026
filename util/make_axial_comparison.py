"""Composite comparison of flat vs axial (S4ND-style 2d/3d/4d) combo variants.

Grouped bar chart: 4 sequence models (LSTM, GRU, Transformer, GPT) x
{flat (embeddings), 2d, 3d, 4d} on the Exp 2 per-combo normalized MSE
(mean +/- seed std, seeds 26/105/115). SSM leaders (Mamba-2, S4) are drawn
as reference lines. Output: graphs/_analysis/axial_comparison.png

Usage: python make_axial_comparison.py
"""
import openpyxl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# parents[1]: this script lives in util/, the repo root is one level up.
ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "results.xlsx"
OUT = ROOT / "graphs" / "_analysis" / "axial_comparison.png"

# Left block = the ACA family (identical axial module, only the temporal
# backbone changes: recurrent / attention / SSM). Right block = native ND SSMs.
# mamba+ACA vs mamba_nd and s4+ACA vs s4nd are matched pairs: same sequence
# model, different way of injecting the grid. mamba_nd_deep is the depth
# ablation of mamba_nd (2 * n_axes layers instead of a flat 4).
MODELS = ["lstm", "gru", "transformer", "gpt", "mamba", "s4", "s4_bidir",
          "s4nd", "mamba_nd", "mamba_nd_deep"]
MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "transformer": "Transformer",
                "gpt": "GPT", "mamba": "Mamba-2\n+ACA", "s4": "S4\n+ACA",
                "s4_bidir": "S4-Bidirectional\n+ACA",
                "s4nd": "S4ND", "mamba_nd": "Mamba-ND",
                "mamba_nd_deep": "Mamba-ND\n(deep)"}
# (column label, variant key in results.xlsx). The recurrent/Transformer
# cross_attention_{2,3,4}d runs now carry the S4ND matched-information
# protocol. Native S4ND and Mamba-ND use their own result labels via REMAP.
CONDS = [
    ("flat", "embeddings"),
    ("2d", "cross_attention_2d"),
    ("3d", "cross_attention_3d"),
    ("4d", "cross_attention_4d"),
]
# Mamba-ND has no own "flat"; its 1-D counterpart is the plain Mamba model.
# Map (display_model, variant_key) -> (results_model, results_variant).
REMAP = {
    ("mamba_nd", "embeddings"): ("mamba", "embeddings"),
    ("mamba_nd", "cross_attention_2d"): ("mamba_nd", "cross_attention_2d"),
    ("mamba_nd", "cross_attention_3d"): ("mamba_nd", "cross_attention_3d"),
    ("mamba_nd", "cross_attention_4d"): ("mamba_nd", "cross_attention_4d"),
    # S4ND: native multidim SSM. flat = 1-D S4; "2d" = the genuine 2-D-by-State
    # (s4nd 2d_state, the paper's reported "2d"); 3d/4d as named.
    ("s4nd", "embeddings"): ("s4", "embeddings"),
    ("s4nd", "cross_attention_2d"): ("s4nd", "2d_state"),
    ("s4nd", "cross_attention_3d"): ("s4nd", "3d"),
    ("s4nd", "cross_attention_4d"): ("s4nd", "4d"),
    # Mamba-ND at coverage-matched depth: its own rows, but the same 1-D Mamba-2
    # flat reference as mamba_nd.
    ("mamba_nd_deep", "embeddings"): ("mamba", "embeddings"),
    # mamba/s4 + ACA need no remap: their flat arms are the plain mamba/s4
    # embeddings runs and their axial arms are stored under their own names.
}
COND_COLORS = {"flat": "#9aa0a6", "2d": "#4285f4", "3d": "#34a853", "4d": "#ea4335"}


def load():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["Exp2_PerCombo"]
    rows = list(ws.iter_rows(values_only=True))
    h = list(rows[0])
    mi, vi = h.index("model"), h.index("variant")
    nm = h.index("norm_mse_combined")
    sd = h.index("norm_mse_std") if "norm_mse_std" in h else None
    d = {}
    for r in rows[1:]:
        if r[mi] is None:
            continue
        d[(r[mi], r[vi])] = (
            r[nm],
            (r[sd] if sd is not None and r[sd] is not None else 0.0),
        )
    return d


def main():
    d = load()
    n_cond = len(CONDS)
    x = np.arange(len(MODELS))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    # Dot plot, not bars: the y-range is narrow (0.049-0.053), so bar heights on
    # a zoomed axis would overstate the differences. Points keep the grouped
    # "flat vs 2d/3d/4d per family" layout without misleading lengths.
    for j, (clabel, vkey) in enumerate(CONDS):
        keys = [REMAP.get((m, vkey), (m, vkey)) for m in MODELS]
        means = [d.get(k, (np.nan, 0))[0] for k in keys]
        errs = [d.get(k, (np.nan, 0))[1] for k in keys]
        xpos = x + (j - (n_cond - 1) / 2) * width
        ax.errorbar(xpos, means, yerr=errs, fmt="o", ms=7, capsize=3,
                    label=clabel, color=COND_COLORS[clabel],
                    markeredgecolor="white", markeredgewidth=0.6,
                    elinewidth=1.0, ls="none")
    # light vertical separators between model groups
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color="#dadce0", lw=0.6, zorder=0)

    # SSM leader reference lines
    mamba = d.get(("mamba", "embeddings"), (None, 0))[0]
    s4 = d.get(("s4", "embeddings"), (None, 0))[0]
    if mamba:
        ax.axhline(mamba, ls="--", lw=1.2, color="#202124")
        ax.text(len(MODELS) - 0.5, mamba, f"  Mamba-2 {mamba:.4f}",
                va="center", ha="left", fontsize=8, color="#202124")
    if s4:
        ax.axhline(s4, ls=":", lw=1.2, color="#5f6368")
        ax.text(len(MODELS) - 0.5, s4, f"  S4 {s4:.4f}",
                va="center", ha="left", fontsize=8, color="#5f6368")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Exp 2 per-combo normalized MSE  (lower = better)")
    ax.set_title("Flat vs. multidimensional structure (2d/3d/4d) across all model families\n"
                 "RNN/Transformer axial inputs use the S4ND-matched identity protocol "
                 "(S4ND/Mamba-ND 'flat' = their 1-D S4/Mamba)")
    ax.set_ylim(0.0485, 0.0535)
    ax.legend(title="combo structure", ncol=4, loc="upper left", frameon=False)
    ax.margins(x=0.02)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
