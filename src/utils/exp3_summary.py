"""Descriptive Exp 3 summaries computed from a complete validated sweep.

No significance or equivalence conclusion follows from a grid range. The
lowest TEST cell is an observed minimum, not a validation-selected setting.
"""
from itertools import product
import math

import pandas as pd


def summarize_sweep(rows, manifest):
    """Return model summaries and marginal means, retaining metric precision.

    Artifact/source/completion validation belongs to the collector. Repeat
    population and numeric checks here so a partial grid cannot become prose.
    Percent ranges are 100 * (maximum - minimum) / minimum, undefined at zero.
    Marginals average the other axis and all seeds with equal weights.
    """
    models = manifest["PAPER_SWEEP_MODELS"].split()
    windows = tuple(map(int, manifest["PAPER_WINDOWS"].split()))
    lags = tuple(map(int, manifest["PAPER_LAGS"].split()))
    seeds = tuple(map(int, manifest["PAPER_SEEDS"].split()))
    expected = set(product(models, windows, lags, seeds))
    keys = [(r["model"], r["window"], r["lag_count"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Exp 3 summary requires the complete canonical grid and seed roster")
    if any(r.get("variant") != "embeddings" for r in rows):
        raise ValueError("Exp 3 summary requires embeddings for every run")
    if len(seeds) < 2:
        raise ValueError("Exp 3 sample seed variability requires at least two seeds")
    losses = [float(r["norm_mse_combined"]) for r in rows]
    if any(not math.isfinite(v) or v < 0 for v in losses):
        raise ValueError("Exp 3 summary requires finite nonnegative normalized MSE")
    frame = pd.DataFrame(rows).assign(norm_mse_combined=losses)
    summaries, marginals = [], []
    for model in models:
        data = frame[frame.model == model]
        cells = data.groupby(["window", "lag_count"])["norm_mse_combined"].agg(["mean", "std"])
        low, high = float(cells["mean"].min()), float(cells["mean"].max())
        # Preserve every exact minimum; rounded display values must never
        # decide ties or which configuration is reported as lowest.
        lowest = cells.index[cells["mean"] == low].tolist()
        summary = {
            "model": model, "metric": "norm_mse_combined",
            "n_cells": len(cells), "n_seeds_per_cell": len(seeds),
            "cell_mean_min": low, "cell_mean_max": high,
            "cell_mean_range": high - low,
            "cell_mean_range_pct_of_min": _range_pct(low, high),
            "lowest_observed_test_cells": "; ".join(f"w{w}_l{lag}" for w, lag in lowest),
            "cell_seed_sample_sd_min": float(cells["std"].min()),
            "cell_seed_sample_sd_max": float(cells["std"].max()),
            "interpretation": "descriptive test sensitivity; not a significance or equivalence test",
        }
        for axis in ("window", "lag_count"):
            marginal = data.groupby(axis)["norm_mse_combined"].mean()
            lo, hi = float(marginal.min()), float(marginal.max())
            summary[f"{axis}_marginal_min"] = lo
            summary[f"{axis}_marginal_max"] = hi
            summary[f"{axis}_marginal_range"] = hi - lo
            summary[f"{axis}_marginal_range_pct_of_min"] = _range_pct(lo, hi)
            marginals.extend({"model": model, "axis": axis, "setting": int(setting),
                              "mean_norm_mse_combined": float(value)}
                             for setting, value in marginal.items())
        summaries.append(summary)
    return summaries, marginals


def _range_pct(low, high):
    return 100 * (high - low) / low if low > 0 else None


def sweep_summary_text(summaries):
    """Render only computed observations; do not turn small ranges into claims."""
    descriptions = []
    for row in summaries:
        pct = row["cell_mean_range_pct_of_min"]
        relative = f"{pct:.6g}% of the minimum" if pct is not None else "undefined because the minimum is zero"
        descriptions.append(
            f"{row['model']}: seed-mean normalized test MSE "
            f"{row['cell_mean_min']:.9g} to {row['cell_mean_max']:.9g}; "
            f"range {row['cell_mean_range']:.9g} ({relative}); "
            f"lowest observed test cells {row['lowest_observed_test_cells']}; "
            f"within-cell sample seed SD {row['cell_seed_sample_sd_min']:.9g} "
            f"to {row['cell_seed_sample_sd_max']:.9g}."
        )
    return " ".join(descriptions) + (
        " These are descriptive observations on the fixed test period. "
        "Neither the grid ranges nor seed variation establish significance, "
        "absence of an effect, or equivalence. Observed test minima are not "
        "validation-selected recommendations. Exp3_Marginals reports each "
        "axis mean over the other axis and all seeds."
    )
