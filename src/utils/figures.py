"""One house style for every figure in the benchmark.

Two things live here so no plotting script can drift from the others:

* ``figure_title`` -- the fixed title format::

      [Group #] on [Model Name[-ND if multidimensional]] With [Embeddings/Onehot]

  where the "N" is a placeholder for the ACTUAL rank and is never printed
  literally: ``Group 3 on GRU-2D With Embeddings``. Deliberately short -- the combo
  identity (state / commodity / flow) lives in the FILENAME, not the title, so
  titles stay one line at a fixed length and can never collide with the axes.

* ``finish_figure`` -- the layout contract: headroom reserved above the data,
  region labels pinned to the axes floor, legend in its own reserved strip,
  and a constrained layout pass. Nothing is drawn where data can reach it.

All values plotted are NORMALIZED (the min-max units the loss is computed in),
never real dollars -- see ``Y_LABEL``.
"""

from __future__ import annotations

# "Time (monthly)", not "Month": the axis IS a datetime axis whose ticks render
# as years, so labelling it "Month" reads as a contradiction under ticks saying
# 2018, 2019, 2020. This states both facts -- the axis is time, the observations
# are monthly (the grid is built at freq="MS"; see src/data/dataloader.py).
X_LABEL = "Time (monthly)"
Y_LABEL = "Normalized value (min-max, train-fit)"
VALUE_LABEL = "Normalized value"
WEIGHT_LABEL = "Normalized weight (SWT)"

# Canonical display names. Anything not listed falls back to a title-cased key.
_MODEL_NAMES = {
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "gpt": "GPT",
    "transformer_linhead": "Transformer (linear head)",
    "gpt_linhead": "GPT (linear head)",
    "mamba": "Mamba2",   # "Mamba-2-2D" would read badly; keep the version tight
    "mamba_nd": "Mamba-ND",
    "s4": "S4",
    "s4_bidir": "S4-Bidirectional",
    "s4nd": "S4ND",
    "dss": "DSS",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "moving_average": "Moving Average",
    "seasonal_naive": "Seasonal Naive",
    "random_walk": "Random Walk",
}

# Variant -> the rank it puts the model at on the (State, Commodity, Flow)
# lattice. The suffix is the ACTUAL dimensionality (GRU-2D, GRU-3D, GRU-4D) --
# "ND" is a placeholder in the spec, never something a title should print.
# s4nd's "2d_state" is the paper's reported 2-D-by-State arm.
_VARIANT_RANK = {
    "cross_attention_2d": "2D", "matched_axial_2d": "2D",
    "cross_attention_3d": "3D", "matched_axial_3d": "3D",
    "cross_attention_4d": "4D", "matched_axial_4d": "4D",
    "2d": "2D", "2d_state": "2D", "3d": "3D", "4d": "4D",
}

# Variants that carry one-hot categoricals; everything else uses embeddings
# (the axial variants append embeddings for the non-structural identities, so
# they read as "Embeddings" too).
_ONEHOT_MARKERS = ("onehot",)


def split_run_name(run_name: str) -> tuple[str, str]:
    """``gru_cross_attention_2d`` -> ``("gru", "cross_attention_2d")``.

    Handles the S4-family convention where the upstream repo writes
    ``s4_s4nd_2d_state`` for what is really model ``s4nd``, variant
    ``2d_state``, and tolerates a trailing ``_seed<N>``.
    """
    name = run_name
    if "_seed" in name:
        name = name.split("_seed")[0]
    if name.startswith("s4_s4nd_"):
        return "s4nd", name[len("s4_s4nd_"):]
    if name.startswith("s4_s4_"):
        variant = name[len("s4_s4_"):]
        return ("s4_bidir" if variant.endswith("_bidir") else "s4",
                variant.removesuffix("_bidir"))
    for key in sorted(_MODEL_NAMES, key=len, reverse=True):
        if name == key:
            return key, ""
        if name.startswith(key + "_"):
            return key, name[len(key) + 1:]
    head, _, tail = name.partition("_")
    return head, tail


def model_display_name(model: str, variant: str = "") -> str:
    """``("gru", "cross_attention_2d")`` -> ``"GRU-2D"``.

    The suffix carries the actual rank, so a model's three multidimensional
    arms stay distinguishable in the title (GRU-2D / GRU-3D / GRU-4D).

    S4ND and Mamba-ND keep their own "ND": that is part of the published method
    name, not a placeholder, so the rank is appended (S4ND-2D) rather than
    substituted. It also keeps the matched pairs apart -- ``S4-2D`` is S4 with
    axial cross-attention, ``S4ND-2D`` is the native ND convolution.
    """
    base = _MODEL_NAMES.get(model, model.replace("_", " ").title())
    rank = _VARIANT_RANK.get(variant)
    return f"{base}-{rank}" if rank else base


def encoder_display_name(variant: str) -> str:
    """``"onehot"`` -> ``"Onehot"``; everything else -> ``"Embeddings"``.

    Spelled out and capitalised -- a title is read, not parsed, and "embd"
    saves eight characters at the cost of making the reader decode it.
    """
    return "Onehot" if variant in _ONEHOT_MARKERS else "Embeddings"


def figure_title(run_name: str, group_index: int) -> str:
    """The fixed title, with the rank substituted for the N::

        Group 12 on GRU-3D With Embeddings
    """
    model, variant = split_run_name(run_name)
    return (f"Group {group_index} on {model_display_name(model, variant)} "
            f"With {encoder_display_name(variant)}")


def finish_figure(fig, ax, *, title: str, xlabel: str = X_LABEL,
                  ylabel: str = Y_LABEL, legend_loc: str = "upper left") -> None:
    """Apply the layout contract.

    Reserves 22% headroom above the data so the legend sits in empty space
    rather than over the series, keeps the title to a single line at a fixed
    size, and runs a tight layout pass so no tick label is ever clipped.
    """
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Headroom: legend and region labels get their own band above the data.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.22 * (high - low))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc=legend_loc, fontsize=8, frameon=True,
                  framealpha=0.92, edgecolor="none", ncol=2, borderpad=0.5)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()


def dual_panel(figsize=(12, 6.6)):
    """Value over weight, sharing one x axis.

    Both targets are min-max normalized, so they already live on the same 0-1
    scale and could be drawn on a single axes -- but four spiky series (actual
    and forecast, twice) cross constantly and the panel stops being readable.
    Stacking keeps each target as legible as a solo plot while the shared x
    axis lines the two up month-for-month, so co-movement is still read off
    vertically. A twin y axis is deliberately NOT used: with both series
    normalized, two differently-scaled axes would imply a difference that is
    not there.
    """
    import matplotlib.pyplot as plt

    # Constrained layout, not tight_layout: with shared axes tight_layout warns
    # that the result "might be incorrect" and can clip the lower panel.
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize,
                             layout="constrained")
    return fig, axes


def finish_dual_panel(fig, axes, *, title: str, xlabel: str = X_LABEL,
                      legend_loc: str = "upper left") -> None:
    """Layout contract for the stacked value/weight figure.

    Title once at the top, legend once in the value panel's reserved headroom,
    y label per panel, x tick labels only under the bottom panel.
    """
    ax_value, ax_weight = axes
    for ax, ylabel in ((ax_value, VALUE_LABEL), (ax_weight, WEIGHT_LABEL)):
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.22 * (high - low))

    ax_value.set_title(title, fontsize=11, pad=10)
    ax_weight.set_xlabel(xlabel, fontsize=9)

    handles, labels = ax_value.get_legend_handles_labels()
    if handles:
        ax_value.legend(handles, labels, loc=legend_loc, fontsize=8,
                        frameon=True, framealpha=0.92, edgecolor="none",
                        ncol=2, borderpad=0.5)
    # Rotate by hand rather than fig.autofmt_xdate(): that calls
    # subplots_adjust, which fights the constrained layout engine.
    for label in ax_weight.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")


def label_split_regions(ax, train_start, val_start, test_start) -> None:
    """Shade and label the train / val / test bands.

    Labels are pinned to the axes floor in axes-fraction coordinates, so they
    never move with the data and can never be overdrawn by a series.
    """
    for x, text, color in (
        (train_start, "train", "#2e7d32"),
        (val_start, "val", "#b8860b"),
        (test_start, "test", "#c62828"),
    ):
        ax.annotate(
            text, xy=(x, 0.02), xycoords=("data", "axes fraction"),
            fontsize=8, color=color, va="bottom", ha="left",
            xytext=(3, 0), textcoords="offset points",
            # A series can dip into the floor band (the test forecast often
            # does), so back the label rather than let a line run through it.
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.78),
        )
