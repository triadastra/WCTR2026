"""Fail-closed regression tests for Experiment 1's aggregate data contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.dataloader import (
    DataConfig,
    TradeDataPipeline,
    EXPORT_VALUE_RAW,
    EXPORT_WEIGHT_RAW,
    IMPORT_VALUE_ALT,
    IMPORT_VALUE_PRI,
    IMPORT_WEIGHT_ALT,
    IMPORT_WEIGHT_PRI,
)
from src.utils.config import get_data_config
from src.utils.provenance import BENCHMARK_PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_standalone_dataloader_protocol_fallback_matches_root_constant():
    import src.data.dataloader as dataloader

    assert dataloader.BENCHMARK_PROTOCOL_VERSION == BENCHMARK_PROTOCOL_VERSION


def _write_tiny_sources(root: Path, *, state: str = "A", import_value: str = "123"):
    imports = root / "imports"
    exports = root / "exports"
    imports.mkdir()
    exports.mkdir()
    pd.DataFrame({
        "State": [state],
        "Commodity": ["01 Food"],
        "Country": ["Region"],
        "Time": ["20-Jan"],
        IMPORT_VALUE_PRI: [import_value],
        IMPORT_VALUE_ALT: ["999"],
        IMPORT_WEIGHT_PRI: ["45"],
        IMPORT_WEIGHT_ALT: ["888"],
    }).to_csv(imports / "A.csv", index=False)
    pd.DataFrame({
        "State": [state],
        "Commodity": ["01 Food"],
        "Country": ["Region"],
        "Time": ["20-Jan"],
        EXPORT_VALUE_RAW: ["10"],
        EXPORT_WEIGHT_RAW: ["2"],
    }).to_csv(exports / "A.csv", index=False)
    return imports, exports


def _tiny_config(imports: Path, exports: Path, **kwargs) -> DataConfig:
    options = dict(imports_dir=str(imports), exports_dir=str(exports))
    options.update(kwargs)
    return DataConfig(**options)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"input_len": 0}, "input_len must be positive"),
        ({"lag_count": -1}, "lag_count must be non-negative"),
        ({"min_nonzero_frac": -0.01}, "between 0 and 1"),
        ({"min_nonzero_frac": 1.01}, "between 0 and 1"),
        ({"start": "2009-01-02"}, "first day of a month"),
        ({"train_end": "2024-01-01"}, "train and validation splits overlap"),
        ({"val_end": "2025-01-01"}, "validation and test splits overlap"),
        ({"test_end": "2025-06-01"}, "test_end must be on or before end"),
        ({"train_target_start": "2024-01-01"}, "between start and train_end"),
    ],
)
def test_data_config_rejects_invalid_or_leaky_calendars(override, message):
    with pytest.raises((TypeError, ValueError), match=message):
        DataConfig(**override)


def test_normalization_method_cannot_lie_about_the_transform():
    with pytest.raises(ValueError, match="supports only"):
        DataConfig(normalize=True, normalization_method="zscore")
    assert DataConfig(normalize=False).normalization_method == "none"


def test_top_level_yaml_splits_are_resolved_into_data_config():
    cfg = {
        "data": {"start": "2019-01-01", "end": "2021-05-01"},
        "splits": {
            "train_end": "2020-12-01",
            "val_start": "2021-01-01",
            "val_end": "2021-02-01",
            "test_start": "2021-03-01",
            "test_end": "2021-05-01",
        },
    }
    resolved = get_data_config(cfg)
    data = DataConfig(**resolved)
    assert data.train_end == "2020-12-01"
    assert data.test_start == "2021-03-01"

    with pytest.raises(ValueError, match="Conflicting data.train_end"):
        get_data_config({
            "data": {"train_end": "2020-01-01"},
            "splits": {"train_end": "2021-01-01"},
        })


def test_month_parser_rejects_ambiguous_and_midmonth_dates():
    pipeline = TradeDataPipeline(DataConfig())
    assert pipeline.parse_month("8-Jan") == pd.Timestamp("2008-01-01")
    assert pipeline.parse_month("2025-05") == pd.Timestamp("2025-05-01")
    assert pipeline.parse_month("2025-05-01") == pd.Timestamp("2025-05-01")
    assert pd.isna(pipeline.parse_month("2025-05-02"))
    assert pd.isna(pipeline.parse_month("05/01/2025"))
    assert pd.isna(pipeline.parse_month("May 2025"))


def test_numeric_validation_rejects_nonfinite_and_negative_trade_values():
    pipeline = TradeDataPipeline(DataConfig())
    with pytest.raises(ValueError, match="Non-finite"):
        pipeline._numeric_clean(pd.Series(["inf"], name="Value"))
    with pytest.raises(ValueError, match="Negative trade"):
        pipeline._numeric_clean(pd.Series(["-1"], name="Weight"))


def test_import_schema_uses_total_vessel_and_fingerprints_exact_bytes(tmp_path):
    imports, exports = _write_tiny_sources(tmp_path)
    pipeline = TradeDataPipeline(_tiny_config(
        imports, exports,
        expected_states=["A"],
        expected_source_start="2020-01-01",
        expected_source_end="2020-01-01",
    ))
    loaded = pipeline.load_data().sort_values("Import/Export").reset_index(drop=True)
    imp = loaded[loaded["Import/Export"] == "Import"].iloc[0]
    assert imp["Value"] == 123.0
    assert imp["Weight"] == 45.0
    assert len(pipeline.source_fingerprint) == 64
    assert len(pipeline.source_files) == 2
    assert {entry["flow"] for entry in pipeline.source_files} == {"imports", "exports"}
    assert all(entry["bytes"] > 0 and len(entry["sha256"]) == 64
               for entry in pipeline.source_files)

    # A value edit with identical series names must invalidate the paper data.
    old = pipeline.source_fingerprint
    export_file = exports / "A.csv"
    export_file.write_text(export_file.read_text().replace(",10,2", ",11,2"))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        TradeDataPipeline(_tiny_config(
            imports, exports, expected_source_fingerprint=old
        )).load_data()


def test_source_domain_rejects_a_different_state_population(tmp_path):
    imports, exports = _write_tiny_sources(tmp_path, state="Unexpected")
    with pytest.raises(ValueError, match="state domain mismatch"):
        TradeDataPipeline(_tiny_config(
            imports, exports, expected_states=["A"]
        )).load_data()


def test_loader_rejects_ambiguous_duplicate_report_grain(tmp_path):
    imports, exports = _write_tiny_sources(tmp_path)
    export_file = exports / "A.csv"
    duplicate = pd.read_csv(export_file, dtype=str)
    duplicate.loc[0, EXPORT_VALUE_RAW] = "11"
    duplicate.to_csv(export_file, mode="a", header=False, index=False)

    with pytest.raises(
        ValueError,
        match=r"ambiguous duplicate report-grain keys: 1 group\(s\), 2 row\(s\)",
    ):
        TradeDataPipeline(_tiny_config(imports, exports)).load_data()


def _aggregate_source() -> pd.DataFrame:
    rows = []
    times = pd.date_range("2020-01-01", periods=12, freq="MS")
    for i, timestamp in enumerate(times, start=1):
        for state, multiplier in (("A", 1.0), ("B", 2.0)):
            value = multiplier * i
            if timestamp >= pd.Timestamp("2020-07-01"):
                value *= 100.0
            rows.append({
                "State": state,
                "Commodity": "Food",
                "Country": "Region",
                "Time": timestamp,
                "Import/Export": "Export",
                "Value": value,
                "Weight": value * 2,
                # Must not be summed into or retained in the model grid.
                "ValuePerUnit": 999999.0,
            })
    return pd.DataFrame(rows)


def test_aggregate_train_scaling_and_target_alignment_are_leakage_free():
    pipeline = TradeDataPipeline(DataConfig(
        start="2020-01-01", end="2020-12-01",
        input_len=2, lag_count=1,
        aggregate=True, filter_sparse=False,
        train_end="2020-06-01",
        val_start="2020-07-01", val_end="2020-09-01",
        test_start="2020-10-01", test_end="2020-12-01",
    ))
    pipeline.df_full = _aggregate_source()
    pipeline.create_splits()

    # Two source series become exactly one aggregate series. The large future
    # values are not allowed into the train-fitted max (June total = 18).
    assert pipeline.kept_combos == [("ALL", "ALL", "ALL")]
    assert len(pipeline.stats["exact"]) == 1
    assert pipeline.stats["exact"].iloc[0]["val_max"] == pytest.approx(18.0)
    assert pipeline.df_train["Value"].max() == pytest.approx(1.0)
    assert pipeline.df_val["Value"].max() > 1.0
    assert "ValuePerUnit" not in pipeline.df_full_grid.columns

    # Metric denominators retain January; model windows begin only after the
    # one-month lag burn-in. Test targets are Oct/Nov/Dec and every input is
    # strictly historical.
    assert pipeline.df_metric_grid["Time"].min() == pd.Timestamp("2020-01-01")
    assert pipeline.df_full_grid["Time"].min() == pd.Timestamp("2020-02-01")
    train, _, test = pipeline.get_dataloaders(
        batch_size=2, num_workers=0, shuffle_train=False
    )
    assert train.drop_last is False
    assert [sample["target_time"] for sample in test.dataset.samples] == [
        pd.Timestamp("2020-10-01"),
        pd.Timestamp("2020-11-01"),
        pd.Timestamp("2020-12-01"),
    ]
    for sample in test.dataset.samples:
        assert np.isfinite(sample["x_numeric"].numpy()).all()


def test_checked_in_data_manifest_matches_the_base_protocol_pin():
    manifest = json.loads((ROOT / "config" / "data_manifest.json").read_text())
    payload = json.dumps(
        {"version": 1, "files": manifest["files"]},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    base = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    assert digest == manifest["source_fingerprint"]
    assert digest == base["data"]["expected_source_fingerprint"]
    assert manifest["states"] == sorted(base["data"]["expected_states"])
    assert manifest["experiment_calendar"] == {
        "grid_start": base["data"]["start"],
        "grid_end": base["data"]["end"],
        **base["splits"],
    }


@pytest.mark.parametrize("variant", ["embeddings", "onehot"])
def test_future_target_mutation_cannot_change_prior_neural_or_tree_inputs(variant):
    """Exercise feature extraction, not just split dates, against a future edit.

    October's target is unavailable for its own forecast; November can use it
    as observed history under the rolling one-step protocol.
    """
    import torch
    from src.models.tabular_common import extract_tabular_xy

    def build(source):
        pipe = TradeDataPipeline(DataConfig(
            start="2020-01-01", end="2020-12-01", input_len=2, lag_count=1,
            aggregate=True, train_end="2020-06-01",
            val_start="2020-07-01", val_end="2020-09-01",
            test_start="2020-10-01", test_end="2020-12-01",
        ))
        pipe.df_full = source
        pipe.create_splits()
        return pipe, pipe.get_dataloaders(batch_size=2, num_workers=0, shuffle_train=False)

    original = _aggregate_source()
    edited = original.copy()
    edited.loc[edited.Time == pd.Timestamp("2020-10-01"), ["Value", "Weight"]] *= 1000
    before, old_loaders = build(original)
    after, new_loaders = build(edited)
    pd.testing.assert_frame_equal(before.stats["exact"], after.stats["exact"], check_exact=True)
    for old, new in zip(old_loaders[:2], new_loaders[:2]):
        assert len(old.dataset) == len(new.dataset)
        for i in range(len(old.dataset)):
            for key in ("x_numeric", "target_value", "target_weight"):
                torch.testing.assert_close(old.dataset[i][key], new.dataset[i][key], rtol=0, atol=0)
    old_test, new_test = old_loaders[2].dataset, new_loaders[2].dataset
    torch.testing.assert_close(old_test[0]["x_numeric"], new_test[0]["x_numeric"], rtol=0, atol=0)
    assert not torch.equal(old_test[0]["target_value"], new_test[0]["target_value"])
    assert not torch.equal(old_test[1]["x_numeric"], new_test[1]["x_numeric"])

    def tree(loader, pipe):
        return extract_tabular_xy(loader, variant=variant,
                                  num_numeric_features=len(pipe.feat_cols),
                                  num_states=1, num_commodities=1, num_flows=1)

    for old, new in zip(old_loaders[:2], new_loaders[:2]):
        old_x, old_y, old_names = tree(old, before)
        new_x, new_y, new_names = tree(new, after)
        np.testing.assert_array_equal(old_x, new_x)
        np.testing.assert_array_equal(old_y, new_y)
        assert old_names == new_names
    old_x, old_y, _ = tree(old_loaders[2], before)
    new_x, new_y, _ = tree(new_loaders[2], after)
    np.testing.assert_array_equal(old_x[0], new_x[0])
    assert not np.array_equal(old_y[0], new_y[0])
    assert not np.array_equal(old_x[1], new_x[1])
