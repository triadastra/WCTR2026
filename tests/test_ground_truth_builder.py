"""Tests for the deterministic, pre-model Experiment 1 target artifact."""

from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from src.data.dataloader import DataConfig, TradeDataPipeline
from src.data.ground_truth import (
    EXP1_CALENDAR,
    EXP1_HS2_CODES,
    EXP1_STATES,
    SOURCE_REPORT_KEY_COLUMNS,
    TARGET_COLUMNS,
    build_ground_truth_manifest,
    build_monthly_target_panel,
    canonical_csv_bytes,
    canonical_manifest_bytes,
    deterministic_gzip,
    validate_exp1_scope,
    validate_source_report_grain,
    verify_artifacts,
)


def _raw_records() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "State": "A", "Commodity": "01 Food", "Country": "X",
            "Time": pd.Timestamp("2020-01-01"), "Import/Export": "Export",
            "Value": 10.0, "Weight": 2.0,
        },
        {
            "State": "A", "Commodity": "01 Food", "Country": "Y",
            "Time": pd.Timestamp("2020-01-01"), "Import/Export": "Export",
            "Value": 5.0, "Weight": 3.0,
        },
        {
            "State": "A", "Commodity": "01 Food", "Country": "X",
            "Time": pd.Timestamp("2020-03-01"), "Import/Export": "Import",
            "Value": 7.0, "Weight": 4.0,
        },
    ])


def _tiny_pipeline() -> TradeDataPipeline:
    return TradeDataPipeline(DataConfig(
        start="2020-01-01", end="2020-03-01",
        train_end="2020-01-01",
        val_start="2020-02-01", val_end="2020-02-01",
        test_start="2020-03-01", test_end="2020-03-01",
    ))


def test_ground_truth_sums_subrecords_and_zero_fills_before_model_transforms():
    panel = build_monthly_target_panel(_tiny_pipeline(), _raw_records())

    assert list(panel.columns) == TARGET_COLUMNS
    assert len(panel) == 6  # two observed series x three complete months
    jan_export = panel[
        (panel["Import/Export"] == "Export")
        & (panel["Time"] == pd.Timestamp("2020-01-01"))
    ].iloc[0]
    assert jan_export["Value"] == 15.0
    assert jan_export["Weight"] == 5.0
    feb = panel[panel["Time"] == pd.Timestamp("2020-02-01")]
    assert (feb[["Value", "Weight"]].to_numpy() == 0).all()
    assert not any(
        column.startswith(("value_lag_", "weight_lag_")) for column in panel
    )
    assert panel.duplicated(["State", "Commodity", "Import/Export", "Time"]).sum() == 0


def test_ambiguous_duplicate_source_report_grain_fails_closed():
    raw = _raw_records()
    conflicting = raw.iloc[[0]].copy()
    conflicting["Value"] = 999.0
    corrupted = pd.concat([raw, conflicting], ignore_index=True)

    with pytest.raises(
        ValueError, match="ambiguous duplicate report-grain keys: 1 group.*2 row"
    ):
        validate_source_report_grain(corrupted)
    with pytest.raises(ValueError, match="Refusing to derive ground truth"):
        build_monthly_target_panel(_tiny_pipeline(), corrupted)

    assert SOURCE_REPORT_KEY_COLUMNS == [
        "State", "Commodity", "Country", "Import/Export", "Time"
    ]


def test_csv_and_gzip_serialization_are_byte_deterministic():
    panel = build_monthly_target_panel(_tiny_pipeline(), _raw_records())
    first_csv = canonical_csv_bytes(panel)
    second_csv = canonical_csv_bytes(panel.sample(frac=1, random_state=7).sort_values(
        ["State", "Commodity", "Import/Export", "Time"], kind="mergesort"
    ))
    assert first_csv == second_csv

    first_gzip = deterministic_gzip(first_csv)
    second_gzip = deterministic_gzip(first_csv)
    assert first_gzip == second_gzip
    assert first_gzip[4:8] == b"\x00\x00\x00\x00"  # gzip mtime
    assert gzip.decompress(first_gzip) == first_csv


def test_manifest_records_preprocessing_boundary_and_detects_tampering(tmp_path):
    pipeline = _tiny_pipeline()
    pipeline.source_fingerprint = "a" * 64
    pipeline.source_files = [
        {"flow": "exports", "name": "A.csv", "bytes": 10, "sha256": "b" * 64}
    ]
    raw = _raw_records()
    panel = build_monthly_target_panel(pipeline, raw)
    csv_bytes = canonical_csv_bytes(panel)
    gzip_bytes = deterministic_gzip(csv_bytes)
    manifest = build_ground_truth_manifest(
        pipeline=pipeline,
        raw=raw,
        panel=panel,
        csv_bytes=csv_bytes,
        gzip_bytes=gzip_bytes,
        artifact_name="targets.csv.gz",
    )
    assert manifest["derivation"] == {
        "aggregation": (
            "sum Value and Weight over distinct Country labels sharing "
            "State, Commodity, Import/Export, Time"
        ),
        "series_population": "series identities observed at least once inside the grid range",
        "missing_months": "zero-filled",
        "sparse_filter_applied": False,
        "normalization_applied": False,
        "lags_applied": False,
        "source_target_groups": 2,
        "source_records_collapsed_beyond_first": 1,
        "zero_filled_target_rows": 4,
        "exp1_aggregate_target": (
            "sum panel Value and Weight over all series independently by Time"
        ),
    }
    assert json.loads(canonical_manifest_bytes(manifest))["artifact"]["sha256"]
    assert manifest["measure_mapping"] == {
        "Export": {
            "Value": "Vessel Total Exports Value ($US)",
            "Weight": "Vessel Total Exports SWT (kg)",
        },
        "Import": {
            "Value": "Vessel Customs Value (Gen) ($US)",
            "Weight": "Vessel SWT (Gen) (kg)",
        },
    }

    output = tmp_path / "targets.csv.gz"
    manifest_path = tmp_path / "manifest.json"
    output.write_bytes(gzip_bytes)
    expected_manifest = canonical_manifest_bytes(manifest)
    manifest_path.write_bytes(expected_manifest)
    verify_artifacts(output, manifest_path, gzip_bytes, expected_manifest)
    output.write_bytes(gzip_bytes + b"tampered")
    with pytest.raises(ValueError, match="differs from pinned source derivation"):
        verify_artifacts(output, manifest_path, gzip_bytes, expected_manifest)


def test_exp1_scope_is_independently_frozen_to_eight_states_and_calendar():
    config = DataConfig(
        expected_source_fingerprint="a" * 64,
        expected_states=list(EXP1_STATES),
        expected_hs2_codes=list(EXP1_HS2_CODES),
        expected_source_start=EXP1_CALENDAR["source_start"],
        expected_source_end=EXP1_CALENDAR["source_end"],
        start=EXP1_CALENDAR["grid_start"],
        end=EXP1_CALENDAR["grid_end"],
        train_end=EXP1_CALENDAR["train_end"],
        val_start=EXP1_CALENDAR["val_start"],
        val_end=EXP1_CALENDAR["val_end"],
        test_start=EXP1_CALENDAR["test_start"],
        test_end=EXP1_CALENDAR["test_end"],
    )
    validate_exp1_scope(config)

    expanded = DataConfig(
        expected_source_fingerprint="a" * 64,
        expected_states=[*EXP1_STATES, "Tennessee"],
        expected_source_start=EXP1_CALENDAR["source_start"],
        expected_source_end=EXP1_CALENDAR["source_end"],
    )
    with pytest.raises(ValueError, match="exactly these eight states"):
        validate_exp1_scope(expanded)

    containerized = DataConfig(
        expected_source_fingerprint="a" * 64,
        expected_states=list(EXP1_STATES),
        expected_source_start=EXP1_CALENDAR["source_start"],
        expected_source_end=EXP1_CALENDAR["source_end"],
        target_measure="containerized_vessel",
    )
    with pytest.raises(ValueError, match="target_measure='total_vessel'"):
        validate_exp1_scope(containerized)
