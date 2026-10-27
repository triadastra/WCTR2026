"""
Single plotting module for inference results.

Every inference run draws its figures through here — no inline matplotlib
in models or scripts (the old research code had ~15 scripts each drawing
slightly different figures; this is the one consolidated path).

All plots read the standardized predictions.csv written by
scripts/inference.py: columns
    State, Commodity, Import/Export, Time,
    pred_value, pred_weight, true_value, true_weight
with values in REAL (denormalized) units.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless servers (the H20 has no display)
import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(times, true, pred, save_path, title="", ylabel="Value"):
    """One true-vs-predicted line plot over time."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(times, true, marker="o", label="actual", color="tab:blue")
    ax.plot(times, pred, marker="x", linestyle="--", label="predicted",
            color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residuals(times, true, pred, save_path, title="", ylabel="Residual"):
    """Residual (predicted - actual) over the test months."""
    resid = pred - true
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(times, resid, marker="o", color="tab:purple", label="residual")
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{ylabel} (pred - actual)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_scatter(true, pred, save_path, title="", label="Value"):
    """Predicted-vs-actual scatter with a y=x reference line."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(true, pred, s=10, alpha=0.4, color="tab:green",
               edgecolors="none")
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="k", linewidth=1, alpha=0.6,
            label="y = x")
    ax.set_title(title)
    ax.set_xlabel(f"actual {label}")
    ax.set_ylabel(f"predicted {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_run(predictions_csv, plots_dir, title="", max_combo_plots=25, aggregate=False):
    """Figure set for one inference run, drawn over the FULL test horizon.

    ``aggregate=True`` — the aggregated single-series test (Exp 1): the data is one
    summed series, so plot it directly (predicted vs actual across the test horizon)
    + residual + scatter. Aggregating is correct *only* here.

    ``aggregate=False`` — the per-combo benchmark (Exp 2): do NOT sum over combos
    (that hides per-series error). Instead write a SEPARATE predicted-vs-actual plot
    PER COMBO over the full test horizon to ``plots/combos/``, plus a global
    predicted-vs-actual scatter for a quick overall view. ``max_combo_plots`` caps
    how many combos are drawn (highest mean value first); -1/None draws every kept
    combo.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(predictions_csv, parse_dates=["Time"])
    key_cols = ["State", "Commodity", "Import/Export"]

    if aggregate:
        # One series ("ALL"): plot it over the test horizon — this IS the series plot.
        agg = df.groupby("Time")[
            ["pred_value", "true_value", "pred_weight", "true_weight"]
        ].sum().reset_index()
        plot_predictions(agg["Time"], agg["true_value"], agg["pred_value"],
                         plots_dir / "aggregate_value.png",
                         title=f"{title} — aggregate value (test horizon)", ylabel="Value ($)")
        plot_predictions(agg["Time"], agg["true_weight"], agg["pred_weight"],
                         plots_dir / "aggregate_weight.png",
                         title=f"{title} — aggregate weight (test horizon)", ylabel="Weight")
        try:
            plot_residuals(agg["Time"], agg["true_value"], agg["pred_value"],
                           plots_dir / "residual_value.png",
                           title=f"{title} — aggregate value residual", ylabel="Value ($)")
            plot_scatter(df["true_value"], df["pred_value"], plots_dir / "scatter_value.png",
                         title=f"{title} — value", label="Value ($)")
        except Exception as e:
            print(f"[warn] diagnostic plots failed: {e}")
        print(f"Plots (aggregate test) written to {plots_dir}")
        return

    # Per-combo benchmark: a SEPARATE plot per combo over the full test horizon.
    ranking = df.groupby(key_cols)["true_value"].mean().sort_values(ascending=False)
    keys = (ranking.index if (max_combo_plots is None or max_combo_plots < 0)
            else ranking.head(max_combo_plots).index)
    combos_dir = plots_dir / "combos"
    combos_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, key in enumerate(keys):
        sub = df[(df["State"] == key[0]) & (df["Commodity"] == key[1])
                 & (df["Import/Export"] == key[2])].sort_values("Time")
        safe = "".join(c if c.isalnum() or c in "_-" else "_"
                       for c in "_".join(str(k) for k in key))[:80]
        plot_predictions(sub["Time"], sub["true_value"], sub["pred_value"],
                         combos_dir / f"{i:03d}_{safe}_value.png",
                         title=f"{title} — {key[0]} / {key[1]} / {key[2]} (test horizon)",
                         ylabel="Value ($)")
        n += 1
    # Global predicted-vs-actual scatter (all combos x test months) — overall view,
    # not a sum-over-combos aggregate.
    try:
        plot_scatter(df["true_value"], df["pred_value"], plots_dir / "scatter_value.png",
                     title=f"{title} — predicted vs actual value", label="Value ($)")
        plot_scatter(df["true_weight"], df["pred_weight"], plots_dir / "scatter_weight.png",
                     title=f"{title} — predicted vs actual weight", label="Weight")
    except Exception as e:
        print(f"[warn] scatter failed: {e}")
    print(f"Plots written to {plots_dir}: {n} per-combo (of {len(ranking)} kept) "
          f"+ scatter — no aggregate (per-combo benchmark)")
