"""Regression tests for the lag burn-in and the sliding-window boundary.

These pin the ONE thing that had no coverage and therefore silently regressed:
how many early months `add_lags` removes, and where the first prediction target
lands as a result.

The contract, stated once:

    a target at time t needs a window [t - input_len, t)
    every row in that window needs its own lag_1..lag_K to be REAL
    the earliest row with real lags is  start + K months
    => the first valid target is  start + K + input_len  months

The `K` months are HISTORY THE FIRST TARGET CONSUMES, not rows to delete.
Deleting `input_len + K` months instead applies the window requirement twice:
once by deletion and again when TradeDataset builds its first window.

All tests run on a synthetic ramp (Value at month index i == i) so a lag's
provenance is checkable by arithmetic, and so nothing here depends on the real
dataset or on the cohort size. Cohort-size assertions live in
tests/test_data_pipeline.py and are deliberately kept separate: time
correctness and population size are different claims and must fail
independently.
"""
import pathlib
import sys

import pandas as pd
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.data.dataloader import (  # noqa: E402
    DataConfig, TradeDataPipeline, KEYS, VALUE_CANON, WEIGHT_CANON,
)

COMBOS = [("StateA", "Corn", "Export"),
          ("StateA", "Corn", "Import"),
          ("StateB", "Soy", "Export")]


def _synthetic(start, n_months):
    """Monthly panel where Value at month index i is exactly i.

    That makes lag provenance a subtraction: value_lag_k at index i must be
    i - k. A fabricated (fillna(0.0)) lag shows up as an exact 0.0 where a
    nonzero index was expected.
    """
    times = pd.date_range(start=start, periods=n_months, freq="MS")
    rows = []
    for state, comm, flow in COMBOS:
        for i, t in enumerate(times):
            rows.append({"State": state, "Commodity": comm,
                         "Import/Export": flow, "Time": t,
                         VALUE_CANON: float(i), WEIGHT_CANON: float(i)})
    return pd.DataFrame(rows)


def _pipe(start, n_months, input_len, lag_count, train_end, val, test):
    times = pd.date_range(start=start, periods=n_months, freq="MS")
    cfg = DataConfig(
        start=start, end=times[-1].strftime("%Y-%m-%d"),
        input_len=input_len, lag_count=lag_count,
        normalize=False,        # keep the ramp readable
        filter_sparse=False,    # cohort selection is a separate concern
        train_end=train_end,
        val_start=val[0], val_end=val[1],
        test_start=test[0], test_end=test[1],
    )
    p = TradeDataPipeline(cfg)
    p.df_full = _synthetic(start, n_months)
    p.create_splits()
    return p


def _first_target(pipe, batch_size=4):
    train, _, _ = pipe.get_dataloaders(batch_size=batch_size, num_workers=0)
    assert len(train.dataset) > 0, "train dataset is empty"
    return min(pd.Timestamp(s["target_time"]) for s in train.dataset.samples)


def _target_months(pipe, batch_size=4):
    train, _, _ = pipe.get_dataloaders(batch_size=batch_size, num_workers=0)
    return len({pd.Timestamp(s["target_time"]) for s in train.dataset.samples})


# --------------------------------------------------------------------------
# 1. Lag provenance: lag_k at time t must be the value at t - k, never a
#    fabricated zero. This is what makes every other assertion meaningful.
# --------------------------------------------------------------------------
def test_every_retained_row_has_real_lags():
    p = _pipe("2009-01-01", 96, input_len=6, lag_count=3,
              train_end="2015-12-01", val=("2016-01-01", "2016-06-01"),
              test=("2016-07-01", "2016-12-01"))
    g = p.df_full_grid
    one = g[(g.State == "StateA") & (g.Commodity == "Corn")
            & (g["Import/Export"] == "Export")].sort_values("Time")
    base = pd.Timestamp("2009-01-01")
    for _, row in one.iterrows():
        idx = (row.Time.year - base.year) * 12 + (row.Time.month - base.month)
        for k in range(1, 4):
            assert row[f"value_lag_{k}"] == pytest.approx(idx - k), (
                f"{row.Time.date()} value_lag_{k} = {row[f'value_lag_{k}']}, "
                f"expected {idx - k}. A value of 0.0 means the row was kept "
                f"despite its lag reaching before config.start."
            )


# --------------------------------------------------------------------------
# 2. The core defect: the burn-in must RESERVE history, not delete it.
# --------------------------------------------------------------------------
def test_first_target_is_start_plus_lag_plus_window():
    """input_len=36, lag_count=12, start 2009-01 -> first target 2013-01.

    Deleting input_len + lag_count months instead pushes it to 2016-01,
    costing 36 target months per series.
    """
    p = _pipe("2009-01-01", 180, input_len=36, lag_count=12,
              train_end="2021-12-01", val=("2022-01-01", "2022-12-01"),
              test=("2023-01-01", "2023-12-01"))
    assert _first_target(p) == pd.Timestamp("2013-01-01")


def test_burn_in_is_month_granular_not_year_granular():
    """lag_count=3, input_len=6 -> first target 2009-10.

    A year-granular cut (`(input_len + lag_count) // 12`) evaluates to 0 years
    here, drops nothing, and lets rows whose lags reach before config.start
    into the first windows.
    """
    p = _pipe("2009-01-01", 96, input_len=6, lag_count=3,
              train_end="2015-12-01", val=("2016-01-01", "2016-06-01"),
              test=("2016-07-01", "2016-12-01"))
    assert _first_target(p) == pd.Timestamp("2009-10-01")


def test_non_january_start_is_honoured():
    """The burn-in must count months from config.start, not from its year."""
    p = _pipe("2009-07-01", 96, input_len=6, lag_count=6,
              train_end="2015-12-01", val=("2016-01-01", "2016-06-01"),
              test=("2016-07-01", "2016-12-01"))
    assert _first_target(p) == pd.Timestamp("2010-07-01")


# --------------------------------------------------------------------------
# 3. Split boundaries. The paper-included loader computed lags per split, so
#    the first rows of val/test carried fabricated zeros. The current pipeline
#    lags the full grid and then filters by target time; this pins that.
# --------------------------------------------------------------------------
def test_val_and_test_rows_do_not_carry_fabricated_lags():
    p = _pipe("2009-01-01", 96, input_len=6, lag_count=3,
              train_end="2015-12-01", val=("2016-01-01", "2016-06-01"),
              test=("2016-07-01", "2016-12-01"))
    base = pd.Timestamp("2009-01-01")
    for name, df in (("val", p.df_val), ("test", p.df_test)):
        one = df[(df.State == "StateA") & (df.Commodity == "Corn")
                 & (df["Import/Export"] == "Export")].sort_values("Time")
        assert len(one) > 0, f"{name} split is empty"
        for _, row in one.iterrows():
            idx = (row.Time.year - base.year) * 12 + (row.Time.month - base.month)
            for k in range(1, 4):
                assert row[f"value_lag_{k}"] == pytest.approx(idx - k), (
                    f"{name} row {row.Time.date()} has a fabricated "
                    f"value_lag_{k}; lags must be computed on the full grid "
                    f"before splitting, not within each split."
                )


# --------------------------------------------------------------------------
# 4. Target-month count. Kept separate from any cohort-size assertion.
# --------------------------------------------------------------------------
def test_train_target_month_count_matches_the_contract():
    p = _pipe("2009-01-01", 180, input_len=36, lag_count=12,
              train_end="2021-12-01", val=("2022-01-01", "2022-12-01"),
              test=("2023-01-01", "2023-12-01"))
    # first target 2013-01, last train target 2021-12 -> 108 months inclusive
    assert _target_months(p) == 108


def test_exp3_target_floor_aligns_different_history_settings():
    common = "2015-01-01"
    short = _pipe("2009-01-01", 180, input_len=6, lag_count=6,
                  train_end="2021-12-01", val=("2022-01-01", "2022-12-01"),
                  test=("2023-01-01", "2023-12-01"))
    long = _pipe("2009-01-01", 180, input_len=36, lag_count=36,
                 train_end="2021-12-01", val=("2022-01-01", "2022-12-01"),
                 test=("2023-01-01", "2023-12-01"))
    short.config.train_target_start = common
    long.config.train_target_start = common
    short_targets = {s["target_time"] for s in short.get_dataloaders(4)[0].dataset.samples}
    long_targets = {s["target_time"] for s in long.get_dataloaders(4)[0].dataset.samples}
    assert short_targets == long_targets
    assert min(short_targets) == pd.Timestamp(common)


def test_train_target_floor_also_applies_to_combo_loader():
    p = _pipe("2009-01-01", 180, input_len=6, lag_count=6,
              train_end="2021-12-01", val=("2022-01-01", "2022-12-01"),
              test=("2023-01-01", "2023-12-01"))
    p.config.train_target_start = "2015-01-01"
    train = p.get_dataloaders(4, combo=True)[0].dataset
    targets = {pd.Timestamp(train.times[i]) for i in train.samples}
    assert min(targets) == pd.Timestamp("2015-01-01")
