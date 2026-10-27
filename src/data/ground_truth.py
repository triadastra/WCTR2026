"""Deterministic construction of the Experiment 1 monthly target panel.

The ground-truth artifact is deliberately upstream of every modelling choice:
it contains raw monthly Value and Weight targets after additive source records
have been collapsed, but before sparse-series filtering, normalization, lag
creation, or train/validation/test sampling.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.census_scope import HS2_CODES
from src.data.dataloader import (
    EXPORT_VALUE_CONTAINERIZED,
    EXPORT_VALUE_TOTAL,
    EXPORT_WEIGHT_CONTAINERIZED,
    EXPORT_WEIGHT_TOTAL,
    IMPORT_VALUE_CONTAINERIZED,
    IMPORT_VALUE_TOTAL,
    IMPORT_WEIGHT_CONTAINERIZED,
    IMPORT_WEIGHT_TOTAL,
    KEYS,
    VALUE_CANON,
    WEIGHT_CANON,
    DataConfig,
    TradeDataPipeline,
)


GROUND_TRUTH_SCHEMA_VERSION = 1
TARGET_COLUMNS = [*KEYS, "Time", VALUE_CANON, WEIGHT_CANON]
TARGET_KEY_COLUMNS = [*KEYS, "Time"]
SOURCE_REPORT_KEY_COLUMNS = [
    "State", "Commodity", "Country", "Import/Export", "Time"
]

# These are intentionally independent of base.yaml.  Editing the experiment
# configuration must not silently expand the population or move its calendar.
EXP1_STATES = (
    "Arkansas",
    "Illinois",
    "Iowa",
    "Louisiana",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Wisconsin",
)
EXP1_HS2_CODES = HS2_CODES
EXP1_CALENDAR = {
    "source_start": "2008-01-01",
    "source_end": "2025-05-01",
    "grid_start": "2009-01-01",
    "grid_end": "2025-05-01",
    "train_end": "2023-12-01",
    "val_start": "2024-01-01",
    "val_end": "2024-12-01",
    "test_start": "2025-01-01",
    "test_end": "2025-05-01",
}


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def validate_exp1_scope(config: DataConfig) -> None:
    """Fail if a config would change the frozen eight-state Exp 1 scope."""
    observed_states = tuple(config.expected_states or ())
    if observed_states != EXP1_STATES:
        raise ValueError(
            "Exp 1 ground truth requires exactly these eight states: "
            f"{list(EXP1_STATES)}; got {list(observed_states)}"
        )
    if config.expected_source_fingerprint is None:
        raise ValueError("Exp 1 ground truth requires a pinned source fingerprint")
    if config.target_measure != "total_vessel":
        raise ValueError("Exp 1 ground truth requires target_measure='total_vessel'")
    if config.commodity_universe != "all_hs_chapters":
        raise ValueError(
            "Exp 1 ground truth requires commodity_universe='all_hs_chapters'"
        )
    if tuple(config.expected_hs2_codes or ()) != EXP1_HS2_CODES:
        raise ValueError(
            "Exp 1 ground truth requires the exact USA Trade Online HS2 "
            f"domain {list(EXP1_HS2_CODES)}; got {config.expected_hs2_codes}"
        )

    observed_calendar = {
        "source_start": config.expected_source_start,
        "source_end": config.expected_source_end,
        "grid_start": config.start,
        "grid_end": config.end,
        "train_end": config.train_end,
        "val_start": config.val_start,
        "val_end": config.val_end,
        "test_start": config.test_start,
        "test_end": config.test_end,
    }
    if observed_calendar != EXP1_CALENDAR:
        raise ValueError(
            "Exp 1 ground-truth calendar is frozen; expected "
            f"{EXP1_CALENDAR}, got {observed_calendar}"
        )


def validate_source_report_grain(raw: pd.DataFrame) -> None:
    """Reject source rows that cannot be uniquely interpreted.

    Country is the finest retained dimension in the checked-in USA Trade
    Online reports. Multiple rows at the same state/commodity/country/flow/
    month grain have no label explaining how they differ, so summing, keeping
    the first, or keeping the last are all unsupported choices. A corrected
    source export must be unique at this grain before it can produce ground
    truth.
    """
    missing = set(SOURCE_REPORT_KEY_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(
            f"Source data is missing report-grain columns: {sorted(missing)}"
        )
    duplicate_rows = raw.duplicated(SOURCE_REPORT_KEY_COLUMNS, keep=False)
    if not duplicate_rows.any():
        return

    duplicate_groups = (
        raw.loc[duplicate_rows]
        .groupby(SOURCE_REPORT_KEY_COLUMNS, dropna=False, observed=True)
        .size()
        .sort_values(ascending=False)
    )
    examples = [
        dict(zip(SOURCE_REPORT_KEY_COLUMNS, key if isinstance(key, tuple) else (key,)))
        for key in duplicate_groups.index[:3]
    ]
    raise ValueError(
        "Source snapshot contains ambiguous duplicate report-grain keys: "
        f"{len(duplicate_groups):,} group(s), {int(duplicate_rows.sum()):,} row(s) "
        f"at {SOURCE_REPORT_KEY_COLUMNS}. Examples: {examples}. Refusing to "
        "derive ground truth until the source report is re-downloaded or an "
        "explicit missing dimension is supplied."
    )


def build_monthly_target_panel(
    pipeline: TradeDataPipeline, raw: pd.DataFrame
) -> pd.DataFrame:
    """Collapse raw records and zero-fill each observed series' monthly grid.

    A *series* is a (State, Commodity, Import/Export) identity that occurs at
    least once inside the configured grid interval.  Every such series gets
    every month in that interval.  This exactly mirrors the model pipeline's
    pre-filter target population; it does not invent all-state Cartesian
    series that never occur in the source snapshot.
    """
    validate_source_report_grain(raw)
    grid = pipeline.build_grid(raw)
    panel = (
        grid[TARGET_COLUMNS]
        .sort_values(TARGET_KEY_COLUMNS, kind="mergesort")
        .reset_index(drop=True)
    )

    if panel.duplicated(TARGET_KEY_COLUMNS).any():
        raise ValueError("Ground-truth panel contains duplicate target keys")
    values = panel[[VALUE_CANON, WEIGHT_CANON]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Ground-truth panel contains invalid target values")

    config_start = pipeline.config.start
    expected_months = pd.date_range(
        config_start, pipeline.config.end, freq="MS"
    )
    observed_months = pd.DatetimeIndex(panel["Time"].unique()).sort_values()
    if not observed_months.equals(expected_months):
        raise ValueError(
            "Ground-truth month domain does not match configured grid: "
            f"{config_start}..{pipeline.config.end}"
        )
    series_sizes = panel.groupby(KEYS, sort=False, observed=True).size()
    if series_sizes.empty or not (series_sizes == len(expected_months)).all():
        raise ValueError("Ground-truth series do not all contain every grid month")
    return panel


def canonical_csv_bytes(panel: pd.DataFrame) -> bytes:
    """Serialize a panel to canonical UTF-8 CSV bytes."""
    missing = set(TARGET_COLUMNS).difference(panel.columns)
    if missing:
        raise ValueError(f"Ground-truth panel is missing columns: {sorted(missing)}")
    canonical = panel.sort_values(TARGET_KEY_COLUMNS, kind="mergesort")
    buffer = io.StringIO(newline="")
    canonical.to_csv(
        buffer,
        columns=TARGET_COLUMNS,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def deterministic_gzip(payload: bytes) -> bytes:
    """Compress bytes without a filename or wall-clock timestamp in header."""
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", fileobj=compressed, mode="wb", compresslevel=9, mtime=0
    ) as stream:
        stream.write(payload)
    return compressed.getvalue()


def _iso_month(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def build_ground_truth_manifest(
    *,
    pipeline: TradeDataPipeline,
    raw: pd.DataFrame,
    panel: pd.DataFrame,
    csv_bytes: bytes,
    gzip_bytes: bytes,
    artifact_name: str,
) -> dict[str, Any]:
    """Build a timestamp-free manifest binding source bytes to target bytes."""
    start = pd.Timestamp(pipeline.config.start)
    end = pd.Timestamp(pipeline.config.end)
    in_scope = raw[raw["Time"].between(start, end, inclusive="both")]
    observed_keys = in_scope[TARGET_KEY_COLUMNS].drop_duplicates()
    series = panel[KEYS].drop_duplicates()
    months = pd.DatetimeIndex(panel["Time"].unique()).sort_values()
    flows = sorted(panel["Import/Export"].unique().tolist())
    states = sorted(panel["State"].unique().tolist())

    if pipeline.config.target_measure == "total_vessel":
        export_value, export_weight = EXPORT_VALUE_TOTAL, EXPORT_WEIGHT_TOTAL
        import_value, import_weight = IMPORT_VALUE_TOTAL, IMPORT_WEIGHT_TOTAL
    else:
        export_value = EXPORT_VALUE_CONTAINERIZED
        export_weight = EXPORT_WEIGHT_CONTAINERIZED
        import_value = IMPORT_VALUE_CONTAINERIZED
        import_weight = IMPORT_WEIGHT_CONTAINERIZED

    return {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "artifact": {
            "filename": artifact_name,
            "media_type": "text/csv",
            "compression": "gzip-level-9-mtime-0",
            "sha256": sha256_bytes(gzip_bytes),
            "canonical_csv_sha256": sha256_bytes(csv_bytes),
            "bytes": len(gzip_bytes),
            "uncompressed_bytes": len(csv_bytes),
        },
        "source": {
            "fingerprint_algorithm": "sha256-manifest-v1",
            "fingerprint": pipeline.source_fingerprint,
            "files": pipeline.source_files,
            "rows_total": int(len(raw)),
            "rows_in_grid": int(len(in_scope)),
            "source_start": _iso_month(raw["Time"].min()),
            "source_end": _iso_month(raw["Time"].max()),
            "report_grain": SOURCE_REPORT_KEY_COLUMNS,
            "duplicate_report_grain_groups": 0,
        },
        "scope": {
            "states": states,
            "state_count": len(states),
            "commodities": int(panel["Commodity"].nunique()),
            "flows": flows,
            "series": int(len(series)),
            "months": int(len(months)),
            "rows": int(len(panel)),
            "grid_start": _iso_month(months.min()),
            "grid_end": _iso_month(months.max()),
            "commodity_universe": pipeline.config.commodity_universe,
            "target_measure": pipeline.config.target_measure,
            "splits": {
                "train_end": _iso_month(pipeline.config.train_end),
                "val_start": _iso_month(pipeline.config.val_start),
                "val_end": _iso_month(pipeline.config.val_end),
                "test_start": _iso_month(pipeline.config.test_start),
                "test_end": _iso_month(pipeline.config.test_end),
            },
        },
        "schema": {
            "columns": TARGET_COLUMNS,
            "key": TARGET_KEY_COLUMNS,
            "units": {VALUE_CANON: "USD", WEIGHT_CANON: "kg"},
        },
        "derivation": {
            "aggregation": (
                "sum Value and Weight over distinct Country labels sharing "
                "State, Commodity, Import/Export, Time"
            ),
            "series_population": (
                "series identities observed at least once inside the grid range"
            ),
            "missing_months": "zero-filled",
            "sparse_filter_applied": False,
            "normalization_applied": False,
            "lags_applied": False,
            "source_target_groups": int(len(observed_keys)),
            "source_records_collapsed_beyond_first": int(
                len(in_scope) - len(observed_keys)
            ),
            "zero_filled_target_rows": int(len(panel) - len(observed_keys)),
            "exp1_aggregate_target": (
                "sum panel Value and Weight over all series independently by Time"
            ),
        },
        "measure_mapping": {
            "Export": {
                VALUE_CANON: export_value,
                WEIGHT_CANON: export_weight,
            },
            "Import": {
                VALUE_CANON: import_value,
                WEIGHT_CANON: import_weight,
            },
        },
    }


def build_ground_truth_bundle(
    config: DataConfig, *, enforce_exp1_scope: bool = True, artifact_name: str
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Return canonical CSV, deterministic gzip, and its manifest."""
    if enforce_exp1_scope:
        validate_exp1_scope(config)
    pipeline = TradeDataPipeline(config)
    raw = pipeline.load_data()
    panel = build_monthly_target_panel(pipeline, raw)
    csv_bytes = canonical_csv_bytes(panel)
    gzip_bytes = deterministic_gzip(csv_bytes)
    manifest = build_ground_truth_manifest(
        pipeline=pipeline,
        raw=raw,
        panel=panel,
        csv_bytes=csv_bytes,
        gzip_bytes=gzip_bytes,
        artifact_name=artifact_name,
    )
    return csv_bytes, gzip_bytes, manifest


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Serialize a ground-truth manifest deterministically."""
    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_artifacts(
    output_path: Path,
    manifest_path: Path,
    gzip_bytes: bytes,
    manifest_bytes: bytes,
) -> None:
    """Write each ground-truth artifact through an atomic file replacement."""
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(f".{output_path.name}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        output_tmp.write_bytes(gzip_bytes)
        manifest_tmp.write_bytes(manifest_bytes)
        output_tmp.replace(output_path)
        manifest_tmp.replace(manifest_path)
    finally:
        output_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def verify_artifacts(
    output_path: Path,
    manifest_path: Path,
    expected_gzip: bytes,
    expected_manifest: bytes,
) -> None:
    """Fail unless checked-in artifacts exactly match a fresh derivation."""
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth artifact: {output_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth manifest: {manifest_path}")
    if output_path.read_bytes() != expected_gzip:
        raise ValueError(
            f"Ground-truth artifact differs from pinned source derivation: {output_path}"
        )
    if manifest_path.read_bytes() != expected_manifest:
        raise ValueError(
            f"Ground-truth manifest differs from pinned source derivation: {manifest_path}"
        )
