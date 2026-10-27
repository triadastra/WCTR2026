"""The figure house style is a contract every plotting script depends on:
one fixed title format and one layout. These pin both."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.utils import figures


@pytest.mark.parametrize(
    ("run_name", "expected"),
    [
        ("lstm_embeddings", "Group 7 on LSTM With Embeddings"),
        ("gpt_onehot", "Group 7 on GPT With Onehot"),
        ("gru_cross_attention_2d", "Group 7 on GRU-2D With Embeddings"),
        ("gru_cross_attention_3d", "Group 7 on GRU-3D With Embeddings"),
        ("gru_cross_attention_4d", "Group 7 on GRU-4D With Embeddings"),
        ("transformer_matched_axial_3d", "Group 7 on Transformer-3D With Embeddings"),
        ("mamba_cross_attention_4d", "Group 7 on Mamba2-4D With Embeddings"),
        ("s4_cross_attention_3d", "Group 7 on S4-3D With Embeddings"),
        ("s4_embeddings", "Group 7 on S4 With Embeddings"),
        ("s4_s4_onehot_seed26", "Group 7 on S4 With Onehot"),
        ("s4_bidir_embeddings", "Group 7 on S4-Bidirectional With Embeddings"),
        ("s4_bidir_cross_attention_3d", "Group 7 on S4-Bidirectional-3D With Embeddings"),
        ("s4_s4_embeddings_bidir_seed26", "Group 7 on S4-Bidirectional With Embeddings"),
        ("s4_s4_onehot_bidir_seed26", "Group 7 on S4-Bidirectional With Onehot"),
        # S4ND / Mamba-ND keep their published "ND" and gain the rank, so the
        # matched pairs stay apart: S4-2D is S4+ACA, S4ND-2D is native ND.
        ("mamba_nd_cross_attention_2d", "Group 7 on Mamba-ND-2D With Embeddings"),
        ("s4_s4nd_2d_state", "Group 7 on S4ND-2D With Embeddings"),
        ("s4_s4nd_4d", "Group 7 on S4ND-4D With Embeddings"),
        # A trailing seed suffix must not leak into the title.
        ("xgboost_onehot_seed26", "Group 7 on XGBoost With Onehot"),
    ],
)
def test_title_format(run_name, expected):
    assert figures.figure_title(run_name, 7) == expected


def test_titles_are_single_line():
    """A second line would push the axes down and break alignment across the
    figure set — the combo identity belongs in the filename, not the title."""
    for run in ("gru_cross_attention_2d", "s4_s4nd_3d", "lstm_onehot"):
        assert "\n" not in figures.figure_title(run, 1)


def test_finish_figure_reserves_headroom_above_the_data():
    """The legend sits in a reserved band, so it can never cover a series."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0.0, 1.0, 0.5], label="actual")
    data_top = 1.0
    figures.finish_figure(fig, ax, title=figures.figure_title("gru_embeddings", 1))
    assert ax.get_ylim()[1] > data_top, "no headroom reserved above the data"
    assert ax.get_ylabel() == figures.Y_LABEL, "y axis must state normalized units"
    plt.close(fig)


def test_dual_panel_shares_one_x_axis_and_titles_once():
    """Value over weight: both normalized to the same 0-1 scale, so stacking
    costs nothing in comparability while keeping each panel readable. The x
    axis is shared so the two line up month-for-month."""
    fig, axes = figures.dual_panel()
    ax_value, ax_weight = axes
    for ax in axes:
        ax.plot([0, 1, 2], [0.0, 1.0, 0.5], label="actual")
    figures.finish_dual_panel(
        fig, axes, title=figures.figure_title("gru_cross_attention_2d", 4)
    )
    assert ax_value.get_shared_x_axes().joined(ax_value, ax_weight)
    assert ax_value.get_title() == "Group 4 on GRU-2D With Embeddings"
    assert ax_weight.get_title() == "", "title belongs on the top panel only"
    assert ax_value.get_ylabel() == figures.VALUE_LABEL
    assert ax_weight.get_ylabel() == figures.WEIGHT_LABEL
    assert ax_value.get_legend() is not None
    assert ax_weight.get_legend() is None, "one legend per figure"
    assert ax_weight.get_xlabel() == figures.X_LABEL
    plt.close(fig)


def test_region_labels_are_backed_so_a_series_cannot_run_through_them():
    """The test forecast routinely dips into the floor band."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0.0, 1.0, 0.5])
    figures.label_split_regions(ax, 0, 1, 2)
    for t in ax.texts:
        assert t.get_bbox_patch() is not None, f"{t.get_text()} has no backing"
    plt.close(fig)


def test_split_region_labels_are_pinned_to_the_axes_floor():
    """Pinned in axes-fraction coords, so they never drift into the series."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0.0, 1.0, 0.5])
    figures.label_split_regions(ax, 0, 1, 2)
    texts = [t for t in ax.texts if t.get_text() in ("train", "val", "test")]
    assert len(texts) == 3
    for t in texts:
        assert t.xycoords == ("data", "axes fraction")
        assert t.xy[1] < 0.1, "labels must sit at the bottom of the axes"
    plt.close(fig)
