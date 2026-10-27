"""Regression tests for DataConfig.tabular_align_targets.

The tree arms build one supervised row per (feature row t-1 -> target t), so
they need 2 months of history where a windowed arm needs `input_len` + 1. Left
alone, the trees therefore fit on targets the neural arms were never offered
(measured on the real data: 153,306 vs 121,176 train targets). The flag clamps
the tree TRAIN targets to the first month the windowed arms can emit.

What these pin, and why each is a separate claim:

  1. default OFF  -- flipping it changes what every tree arm trains on, so it
     must never turn itself on.
  2. ON aligns the tree train target set to the windowed one EXACTLY, not
     approximately.
  3. features are untouched -- this clamps which targets are fit, not what the
     model sees per row. A tree row is still t-1 plus lag_count lags; the flag
     must not silently hand trees a window.
  4. val/test are never touched, in either state. They already carry an
     explicit target_time_start, so evaluation must be bit-identical.

Synthetic ramp panel throughout, so nothing here depends on the real dataset
or the cohort size.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.data.dataloader import (  # noqa: E402
    DataConfig, TradeDataPipeline, VALUE_CANON, WEIGHT_CANON,
)
from src.models.tabular_common import extract_tabular_xy  # noqa: E402

COMBOS = [("StateA", "Corn", "Export"),
          ("StateA", "Corn", "Import"),
          ("StateB", "Soy", "Export")]

START = "2009-01-01"
N_MONTHS = 96
INPUT_LEN = 12
LAG_COUNT = 6
TRAIN_END = "2014-12-01"
VAL = ("2015-01-01", "2015-06-01")
TEST = ("2015-07-01", "2015-12-01")


def _synthetic():
    times = pd.date_range(start=START, periods=N_MONTHS, freq="MS")
    rows = []
    for state, comm, flow in COMBOS:
        for i, t in enumerate(times):
            rows.append({"State": state, "Commodity": comm,
                         "Import/Export": flow, "Time": t,
                         VALUE_CANON: float(i + 1), WEIGHT_CANON: float(i + 1)})
    return pd.DataFrame(rows)


def _pipe(align):
    times = pd.date_range(start=START, periods=N_MONTHS, freq="MS")
    cfg = DataConfig(
        start=START, end=times[-1].strftime("%Y-%m-%d"),
        input_len=INPUT_LEN, lag_count=LAG_COUNT,
        normalize=False, filter_sparse=False,
        tabular_align_targets=align,
        train_end=TRAIN_END, val_start=VAL[0], val_end=VAL[1],
        test_start=TEST[0], test_end=TEST[1],
    )
    p = TradeDataPipeline(cfg)
    p.df_full = _synthetic()
    p.create_splits()
    return p


def _loaders(pipe):
    return pipe.get_dataloaders(batch_size=4, num_workers=0, shuffle_train=False)


def _tabular(pipe, loader):
    return extract_tabular_xy(
        loader,
        variant="embeddings",
        num_numeric_features=len(pipe.feat_cols),
        num_states=len(pipe.state2id),
        num_commodities=len(pipe.comm2id),
        num_flows=len(pipe.flow2id),
    )


def _neural_targets(loader):
    return {pd.Timestamp(s["target_time"]) for s in loader.dataset.samples}


def test_flag_defaults_off():
    """Flipping it invalidates every existing tree run, so OFF is the default."""
    assert DataConfig().tabular_align_targets is False


def test_off_lets_trees_start_earlier_than_the_windowed_arms():
    pipe = _pipe(align=False)
    train, _, _ = _loaders(pipe)
    X_tree, _, _ = _tabular(pipe, train)
    n_neural = len(train.dataset)

    assert X_tree.shape[0] > n_neural, (
        "without the flag the tree extraction should out-count the windowed "
        "arms; if it does not, the asymmetry this flag exists for is gone"
    )
    assert getattr(train.dataset, "tabular_target_time_start", None) is None


def test_on_matches_the_windowed_train_target_set_exactly():
    pipe = _pipe(align=True)
    train, _, _ = _loaders(pipe)
    X_tree, y_tree, _ = _tabular(pipe, train)

    assert X_tree.shape[0] == len(train.dataset)
    assert y_tree.shape[0] == len(train.dataset)
    assert train.dataset.tabular_target_time_start == min(_neural_targets(train))


def test_on_does_not_change_the_feature_row():
    """Alignment clamps targets, not features: same columns, same width, and
    the surviving rows are unchanged."""
    off, on = _pipe(align=False), _pipe(align=True)
    tr_off, _, _ = _loaders(off)
    tr_on, _, _ = _loaders(on)

    X_off, y_off, names_off = _tabular(off, tr_off)
    X_on, y_on, names_on = _tabular(on, tr_on)

    assert names_off == names_on
    assert X_off.shape[1] == X_on.shape[1]

    # Rows are concatenated group by group in a stable order, so the aligned
    # matrix is each native group block with its earliest rows dropped -- not
    # one contiguous slice of the whole matrix.
    n_groups = len(COMBOS)
    assert X_off.shape[0] % n_groups == 0 and X_on.shape[0] % n_groups == 0
    k_off, k_on = X_off.shape[0] // n_groups, X_on.shape[0] // n_groups
    assert k_on < k_off, "alignment should drop the earliest targets in each group"

    for g in range(n_groups):
        native_x = X_off[g * k_off:(g + 1) * k_off]
        native_y = y_off[g * k_off:(g + 1) * k_off]
        aligned_x = X_on[g * k_on:(g + 1) * k_on]
        aligned_y = y_on[g * k_on:(g + 1) * k_on]
        # suffix, i.e. the same late targets with identical feature rows
        assert np.allclose(native_x[-k_on:], aligned_x, equal_nan=True), f"group {g} features differ"
        assert np.allclose(native_y[-k_on:], aligned_y, equal_nan=True), f"group {g} targets differ"


@pytest.mark.parametrize("align", [False, True])
def test_val_and_test_are_never_touched(align):
    """Evaluation must be bit-identical in both states."""
    base, pipe = _pipe(align=False), _pipe(align=align)
    _, va_b, te_b = _loaders(base)
    _, va_p, te_p = _loaders(pipe)

    for loader_b, loader_p in ((va_b, va_p), (te_b, te_p)):
        Xb, yb, _ = _tabular(base, loader_b)
        Xp, yp, _ = _tabular(pipe, loader_p)
        assert Xb.shape == Xp.shape
        assert np.allclose(Xb, Xp, equal_nan=True)
        assert np.allclose(yb, yp, equal_nan=True)
        assert getattr(loader_p.dataset, "tabular_target_time_start", None) is None


def test_windowed_arms_are_unaffected_by_the_flag():
    """The flag is tree-only; the neural sample set must not move."""
    off, on = _pipe(align=False), _pipe(align=True)
    tr_off, va_off, te_off = _loaders(off)
    tr_on, va_on, te_on = _loaders(on)

    assert len(tr_off.dataset) == len(tr_on.dataset)
    assert _neural_targets(tr_off) == _neural_targets(tr_on)
    assert len(va_off.dataset) == len(va_on.dataset)
    assert len(te_off.dataset) == len(te_on.dataset)
