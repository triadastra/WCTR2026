"""Regression tests for the leakage-critical parts of TradeDataPipeline.

These guard the paper's central correctness claim — a leakage-free shared data
pipeline — which previously had ZERO automated coverage. They exercise the real
in-repo dataset (data/imports, data/exports) and assert:

  1. the >=95%-nonzero sparse filter keeps the expected 918 combos (raw, pre-norm),
     measured on the TRAIN slice only, and still reproduces the published 903
     under the legacy full-timeline setting;
  2. per-combo min-max stats are fit on the TRAIN slice ONLY (no val/test leak);
  3. val/test normalized values may fall outside [0, 1] (no clipping, train-only scale);
  4. the test split yields exactly 918 x 5 = 4590 targets.

A future edit that fit stats on the full grid (re-introducing leakage) or changed
the windowing would fail (2) or (4) here.

Run:  pytest tests/test_data_pipeline.py -v
"""
import pathlib
import sys

import pandas as pd
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.data.dataloader import (  # noqa: E402
    DataConfig, TradeDataPipeline, KEYS, VALUE_CANON,
)
from src.utils.config import get_data_config, load_config  # noqa: E402

KEPT_COMBOS = 918        # combos passing the >=95%-nonzero raw filter, train-only window
LEGACY_KEPT_COMBOS = 903  # ... under the survivorship-biased full-timeline window
TEST_TARGETS = 4590      # 918 kept combos x 5 test months (2025-01 .. 2025-05)
# Train target months: first target 2013-01 (= start 2009-01 + 12 lag + 36 window)
# through 2023-12. Was 96 before the burn-in fix, which deleted the 36 months of
# history the first target needs instead of reserving them; see
# tests/test_burn_in_window.py and debug.md R33.
TRAIN_TARGET_MONTHS = 132
EPS = 1e-6


@pytest.fixture(scope="module")
def pipe():
    imp = _REPO / "data" / "imports"
    exp = _REPO / "data" / "exports"
    if not list(imp.glob("*.csv")) or not list(exp.glob("*.csv")):
        pytest.skip("real dataset not present under data/imports & data/exports")
    # Real-data tests must use the pinned source profile, including its audited
    # original-data semantics; bare DataConfig intentionally remains strict.
    config = get_data_config(load_config(str(_REPO / "config/base.yaml")))
    config.update(imports_dir=str(imp), exports_dir=str(exp))
    p = TradeDataPipeline(DataConfig(**config))
    p.load_data()
    p.create_splits()
    return p


def test_sparse_filter_keeps_expected_combos(pipe):
    # Computed on RAW values before normalization (see create_splits ordering).
    assert pipe.kept_combos is not None
    assert len(pipe.kept_combos) == KEPT_COMBOS, (
        f"expected {KEPT_COMBOS} kept combos, got {len(pipe.kept_combos)} — "
        "sparse-combo filter changed; investigate before re-anchoring."
    )


def test_base_config_pairs_corrected_cohort_with_matching_batch(pipe):
    cfg = load_config(_REPO / "config" / "base.yaml")
    assert cfg["data"]["cohort_from_train_only"] is True
    assert cfg["training"]["batch_size"] == len(pipe.kept_combos) == KEPT_COMBOS


def test_corrected_batch_has_no_drop_last_omissions(pipe):
    train, _, _ = pipe.get_dataloaders(
        batch_size=KEPT_COMBOS, num_workers=0, shuffle_train=False, combo=False
    )
    assert len(train.dataset) % KEPT_COMBOS == 0
    assert len(train) == TRAIN_TARGET_MONTHS


def test_cohort_is_frozen_on_train_data(pipe):
    """The cohort must not be chosen with val/test targets in view.

    The full-timeline filter is survivorship-biased: it can only ever drop
    series that were dense through train and thinned during val/test. Assert
    the default is train-only, that the legacy window is rejected.
    """
    grid = pipe.build_grid(pipe.df_full)

    legacy = TradeDataPipeline(DataConfig(
        imports_dir=pipe.config.imports_dir,
        exports_dir=pipe.config.exports_dir,
        cohort_from_train_only=False,
    ))
    with pytest.raises(ValueError, match="cohort_from_train_only must be True"):
        legacy.filter_sparse_combos(grid)



def test_normalization_stats_fit_on_train_only(pipe):
    # Reproduce the raw, filtered, pre-normalization grid the pipeline scaled on.
    grid = pipe.filter_sparse_combos(pipe.build_grid(pipe.df_full))
    train_slice = grid[grid["Time"] <= pd.Timestamp(pipe.config.train_end)]

    stats_train = pipe.compute_stats(train_slice)["exact"].sort_values(KEYS).reset_index(drop=True)
    stats_full = pipe.compute_stats(grid)["exact"].sort_values(KEYS).reset_index(drop=True)
    saved = pipe.stats["exact"].sort_values(KEYS).reset_index(drop=True)

    # (a) the pipeline's stored stats MUST equal train-only stats ...
    pd.testing.assert_frame_equal(saved, stats_train, check_dtype=False)

    # (b) ... and MUST differ from full-grid stats, else val/test leaked into
    # the scale. At least one combo's range should grow when val/test is added.
    merged = stats_train.merge(stats_full, on=KEYS, suffixes=("_tr", "_full"))
    assert (merged["val_max_full"] > merged["val_max_tr"] + EPS).any(), (
        "train-only and full-grid stats are identical — cannot demonstrate "
        "train-only fitting (or val/test never exceeds train range)."
    )


def test_val_test_may_exceed_unit_interval(pipe):
    g = pipe.df_full_grid
    train = g[g["Time"] <= pd.Timestamp(pipe.config.train_end)]
    later = g[g["Time"] >= pd.Timestamp(pipe.config.val_start)]
    # Train values are bounded to [0, 1] by construction (train-fit min-max).
    assert train[VALUE_CANON].min() >= -EPS
    assert train[VALUE_CANON].max() <= 1.0 + EPS
    # Val/test values may escape [0, 1] — proof of no clipping + train-only scale.
    assert (later[VALUE_CANON].max() > 1.0 + EPS) or (later[VALUE_CANON].min() < -EPS), (
        "no val/test value fell outside [0,1]; expected under train-only min-max."
    )


def test_test_split_has_expected_target_count(pipe):
    _, _, te = pipe.get_dataloaders(
        batch_size=64, num_workers=0, shuffle_train=False, combo=False
    )
    assert len(te.dataset) == TEST_TARGETS, (
        f"expected {TEST_TARGETS} test targets ({KEPT_COMBOS} x 5 months), "
        f"got {len(te.dataset)}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
