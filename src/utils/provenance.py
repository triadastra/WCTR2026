"""Protocol provenance checks for experiment artifacts.

The corrected per-combination protocol freezes the cohort on the training
period.  That produces 918 series and five test targets per series.  Keeping
this check in one place prevents pre-correction 903-series artifacts from
being mistaken for completed runs by launchers or result builders.
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.inference.metrics import (
    DIRECT_METRIC_NAMES,
    REQUIRED_EVALUATION_METRICS,
    compute_evaluation_metrics,
)


CORRECTED_NUM_COMBOS = 918
CORRECTED_TEST_MONTHS = 5
CORRECTED_TEST_TARGETS = CORRECTED_NUM_COMBOS * CORRECTED_TEST_MONTHS
PROTOCOL_VERSION = "train-only-cohort-v1"
from .benchmark_protocol import BENCHMARK_PROTOCOL_VERSION
CORRECTED_TEST_DATES = {
    date(2025, month, 1) for month in range(1, CORRECTED_TEST_MONTHS + 1)
}


def cohort_provenance(num_combos: int, train_only: bool) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "cohort_from_train_only": bool(train_only),
        "num_combos": int(num_combos),
    }


def data_protocol(input_len: int, lag_count: int, train_target_start: Any) -> dict[str, Any]:
    """Training-history settings that determine which supervised rows exist."""
    return {
        "input_len": int(input_len),
        "lag_count": int(lag_count),
        "train_target_start": (
            None if train_target_start in (None, "") else str(train_target_start)[:10]
        ),
    }


def validate_required_metric_suite(metrics: Any) -> tuple[bool, str]:
    """Require one finite value for every canonical benchmark metric."""
    if not isinstance(metrics, Mapping):
        return False, "missing metrics payload"
    missing = [name for name in REQUIRED_EVALUATION_METRICS if name not in metrics]
    if missing:
        return False, f"missing required metrics {missing}"
    for name in REQUIRED_EVALUATION_METRICS:
        value = metrics[name]
        if isinstance(value, bool):
            return False, f"metric {name} is not numeric"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False, f"metric {name} is not numeric"
        if not math.isfinite(numeric):
            return False, f"metric {name} is not finite"
    return True, "current"


def corrected_run_contract_artifact(payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate the self-contained contract embedded in a result payload."""
    contract = payload.get("run_contract")
    if not isinstance(contract, Mapping):
        return False, "missing run contract"
    if contract.get("run_contract_version") != 1:
        return False, "unknown or legacy run contract"
    if contract.get("benchmark_protocol_version") != BENCHMARK_PROTOCOL_VERSION:
        return False, "run contract predates the current benchmark protocol"
    expected_identity = {
        "model": payload.get("model"),
        "variant": payload.get("variant"),
        "seed": payload.get("seed"),
        "aggregate": payload.get("aggregate"),
    }
    if contract.get("identity") != expected_identity:
        return False, "run contract identity does not match metrics identity"
    source_fingerprint = payload.get("source_fingerprint")
    if not isinstance(source_fingerprint, str) or re.fullmatch(
        r"[0-9a-f]{64}", source_fingerprint
    ) is None:
        return False, "missing or invalid source-data fingerprint"
    if contract.get("source_fingerprint") != source_fingerprint:
        return False, "run contract source fingerprint does not match metrics"
    source_domain = payload.get("source_domain")
    if not isinstance(source_domain, Mapping) or not source_domain:
        return False, "missing source state/calendar domain"
    if contract.get("source_domain") != source_domain:
        return False, "run contract source domain does not match metrics"
    for field in ("model_config", "data_config", "training_config", "selection"):
        if not isinstance(contract.get(field), Mapping):
            return False, f"run contract has no {field} mapping"
    return True, "current"


def corrected_per_combo_artifact(payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Return whether a metrics payload follows the corrected test protocol."""
    if payload.get("aggregate") is not False:
        return False, "artifact is not explicitly identified as a per-combination run"
    current, reason = corrected_run_contract_artifact(payload)
    if not current:
        return current, reason
    cohort = payload.get("cohort")
    if not isinstance(cohort, Mapping):
        return False, "missing cohort provenance"
    if cohort.get("protocol_version") != PROTOCOL_VERSION:
        return False, "unknown or legacy cohort protocol"
    if cohort.get("benchmark_protocol_version") != BENCHMARK_PROTOCOL_VERSION:
        return False, "artifact predates the current fair-comparison protocol"
    protocol = payload.get("data_protocol")
    if not isinstance(protocol, Mapping):
        return False, "missing training-history protocol"
    try:
        if int(protocol["input_len"]) <= 0 or int(protocol["lag_count"]) < 0:
            return False, "invalid training-history protocol"
    except (KeyError, TypeError, ValueError):
        return False, "invalid training-history protocol"
    if "train_target_start" not in protocol:
        return False, "invalid training-history protocol"
    if cohort.get("cohort_from_train_only") is not True:
        return False, "cohort was not selected from training data only"
    try:
        num_combos = int(cohort.get("num_combos", -1))
        n_samples = int(payload.get("n_samples", -1))
    except (TypeError, ValueError):
        return False, "invalid cohort or target count"
    if num_combos != CORRECTED_NUM_COMBOS:
        return False, f"expected {CORRECTED_NUM_COMBOS} combinations"
    if n_samples != CORRECTED_TEST_TARGETS:
        return False, f"expected {CORRECTED_TEST_TARGETS} test targets"
    if payload.get("mode") != "one_step_normalized":
        return False, "evaluation was not run in normalized one-step mode"
    if payload.get("prediction_units") != "normalized":
        return False, "prediction columns are not declared normalized"
    if payload.get("scale_dependent_metric_units") != "normalized":
        return False, "scale-dependent metrics are not declared normalized"
    if payload.get("percentage_metric_units") != "real":
        return False, "percentage metrics are not declared in physical units"
    metrics = payload.get("metrics", payload)
    return validate_required_metric_suite(metrics)


def corrected_aggregate_artifact(payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Return whether a single-series aggregate run uses the current protocol."""
    if payload.get("aggregate") is not True:
        return False, "artifact is not explicitly identified as an aggregate run"
    current, reason = corrected_run_contract_artifact(payload)
    if not current:
        return current, reason
    cohort = payload.get("cohort")
    if not isinstance(cohort, Mapping):
        return False, "missing cohort provenance"
    if cohort.get("protocol_version") != PROTOCOL_VERSION:
        return False, "unknown or legacy cohort protocol"
    if cohort.get("benchmark_protocol_version") != BENCHMARK_PROTOCOL_VERSION:
        return False, "artifact predates the current fair-comparison protocol"
    if cohort.get("cohort_from_train_only") is not True:
        return False, "cohort was not selected from training data only"
    try:
        if int(cohort.get("num_combos", -1)) != 1:
            return False, "aggregate artifact must contain one series"
        if int(payload.get("n_samples", -1)) != CORRECTED_TEST_MONTHS:
            return False, f"expected {CORRECTED_TEST_MONTHS} aggregate test targets"
    except (TypeError, ValueError):
        return False, "invalid aggregate series or target count"
    if payload.get("mode") != "one_step_normalized":
        return False, "evaluation was not run in normalized one-step mode"
    if payload.get("prediction_units") != "normalized":
        return False, "prediction columns are not declared normalized"
    if payload.get("scale_dependent_metric_units") != "normalized":
        return False, "scale-dependent metrics are not declared normalized"
    if payload.get("percentage_metric_units") != "real":
        return False, "percentage metrics are not declared in physical units"
    protocol = payload.get("data_protocol")
    if not isinstance(protocol, Mapping):
        return False, "missing training-history protocol"
    try:
        if int(protocol["input_len"]) <= 0 or int(protocol["lag_count"]) < 0:
            return False, "invalid training-history protocol"
    except (KeyError, TypeError, ValueError):
        return False, "invalid training-history protocol"
    if "train_target_start" not in protocol:
        return False, "invalid training-history protocol"
    metrics = payload.get("metrics", payload)
    return validate_required_metric_suite(metrics)


PREDICTION_COLUMNS = {
    "State", "Commodity", "Import/Export", "Time",
    "pred_value", "pred_weight", "true_value", "true_weight",
    "pred_value_real", "pred_weight_real", "true_value_real", "true_weight_real",
}


def corrected_prediction_artifact(
    payload: Mapping[str, Any], metrics_path: Path,
    *, expected_num_combos: int = CORRECTED_NUM_COMBOS,
    expected_months: int = CORRECTED_TEST_MONTHS,
    expected_dates: set[date] | frozenset[date] = frozenset(CORRECTED_TEST_DATES),
    expected_combos: set[tuple[str, str, str]] | frozenset[tuple[str, str, str]] | None = None,
    verify_scaled_metrics: bool = False,
) -> tuple[bool, str]:
    """Validate the normalized predictions belonging to a metrics artifact."""
    metrics_path = Path(metrics_path)
    run_dir = metrics_path.parent
    if payload.get("prediction_units") != "normalized":
        return False, "prediction columns are not declared normalized"
    if payload.get("scale_dependent_metric_units") != "normalized":
        return False, "scale-dependent metrics are not declared normalized"
    if payload.get("percentage_metric_units") != "real":
        return False, "percentage metrics are not declared in physical units"
    prediction_name = payload.get("predictions_file")
    if prediction_name not in {"predictions.csv", "predictions_norm.csv"}:
        return False, "missing normalized predictions provenance"
    prediction_path = run_dir / prediction_name
    if not prediction_path.is_file():
        return False, f"missing predictions artifact {prediction_name}"

    try:
        with prediction_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not PREDICTION_COLUMNS.issubset(reader.fieldnames or []):
                return False, f"invalid prediction columns in {prediction_path.name}"
            rows = 0
            identities = set()
            combo_counts = {}
            combo_dates = {}
            predictions_normalized = []
            targets_normalized = []
            predictions_real = []
            targets_real = []
            prediction_keys = []
            prediction_identities = []
            for row in reader:
                rows += 1
                combo = (
                    row["State"], row["Commodity"], row["Import/Export"]
                )
                prediction_keys.append(combo)
                try:
                    target_date = date.fromisoformat(row["Time"][:10])
                    values = [
                        float(row[column]) for column in (
                            "pred_value", "pred_weight", "true_value", "true_weight"
                        )
                    ]
                    real_values = [
                        float(row[column]) for column in (
                            "pred_value_real", "pred_weight_real",
                            "true_value_real", "true_weight_real",
                        )
                    ]
                except (TypeError, ValueError):
                    return False, f"invalid prediction row in {prediction_path.name}"
                if not all(math.isfinite(value) for value in values + real_values):
                    return False, f"non-finite prediction row in {prediction_path.name}"
                predictions_normalized.append(values[:2])
                targets_normalized.append(values[2:])
                predictions_real.append(real_values[:2])
                targets_real.append(real_values[2:])
                identity = combo + (target_date,)
                prediction_identities.append(combo + (target_date.isoformat(),))
                if identity in identities:
                    return False, (
                        f"duplicate prediction target in {prediction_path.name}: "
                        f"{identity}"
                    )
                identities.add(identity)
                combo_counts[combo] = combo_counts.get(combo, 0) + 1
                combo_dates.setdefault(combo, set()).add(target_date)
    except (OSError, csv.Error, UnicodeError) as exc:
        return False, f"unreadable predictions artifact ({exc})"

    expected = int(payload.get("n_samples", -1))
    if rows != expected:
        return False, (
            f"{prediction_path.name} has {rows} rows; expected {expected}"
        )
    if len(combo_counts) != expected_num_combos:
        return False, (
            f"{prediction_path.name} has {len(combo_counts)} combinations; "
            f"expected {expected_num_combos}"
        )
    if expected_combos is not None and set(combo_counts) != set(expected_combos):
        return False, (
            f"{prediction_path.name} contains the wrong series identities; "
            f"expected {sorted(expected_combos)!r}"
        )
    if set(combo_counts.values()) != {expected_months}:
        return False, (
            f"{prediction_path.name} does not contain exactly "
            f"{expected_months} targets per combination"
        )
    if any(dates != expected_dates for dates in combo_dates.values()):
        return False, (
            f"{prediction_path.name} does not match the corrected test horizon"
        )
    # Bind the full declared direct metric suite to the prediction file.
    # Previously only normalized MSE was checked, so MAPE/sMAPE (and every
    # other reporting metric) could be edited independently of the prediction
    # bytes and still reach the paper workbook.
    metrics = payload.get("metrics", payload)
    current, reason = validate_required_metric_suite(metrics)
    if not current:
        return current, reason
    recomputed_direct = compute_evaluation_metrics(
        np.asarray(predictions_normalized, dtype=np.float64),
        np.asarray(targets_normalized, dtype=np.float64),
        np.asarray(predictions_real, dtype=np.float64),
        np.asarray(targets_real, dtype=np.float64),
    )
    for name in ("norm_mse_combined", *DIRECT_METRIC_NAMES):
        declared = float(metrics[name])
        recomputed = float(recomputed_direct[name])
        if not math.isfinite(recomputed):
            return False, (
                f"{prediction_path.name} produces non-finite metric {name}"
            )
        if not math.isclose(declared, recomputed, rel_tol=1e-6, abs_tol=1e-8):
            return False, (
                f"{prediction_path.name} metric {name}={recomputed:.12g} does "
                f"not match metrics.json value {declared:.12g}"
            )
    if verify_scaled_metrics:
        from .scaled_metric_validation import source_scaled_metrics
        try:
            scaled = source_scaled_metrics(payload, predictions_normalized,
                                           targets_normalized, prediction_keys,
                                           identities=prediction_identities,
                                           predictions_real=predictions_real,
                                           targets_real=targets_real)
            for name, expected_value in scaled.items():
                if not math.isclose(float(metrics[name]), expected_value, rel_tol=1e-6, abs_tol=1e-8):
                    return False, f"metric {name} differs from pinned source seasonal scaling"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return False, f"cannot validate source seasonal metrics: {exc}"
    return True, "current"


def assert_corrected_runs(runs_dir, *, what="analysis", require_predictions=True):
    """Raise unless every run under ``runs_dir`` follows the corrected protocol.

    One call is the whole gate, so a consumer of ``outputs/full`` cannot forget
    half of it. Every script that turns run artifacts into a reported number or
    a paper figure must call this first: the pre-correction artifacts were
    evaluated on a cohort chosen with val/test availability in view (903 lanes,
    4,515 targets), and a figure built from them is a figure built on
    survivorship bias.

    Returns the list of validated run directories so the caller can iterate.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"{runs_dir} does not exist")
    import json as _json

    validated, checked = [], 0
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        metrics_path = d / "metrics.json"
        if not metrics_path.is_file():
            continue
        checked += 1
        try:
            payload = _json.loads(metrics_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Refusing {what}: {d.name} has unreadable metrics ({exc})")
        current, reason = corrected_per_combo_artifact(payload)
        if not current:
            raise RuntimeError(
                f"Refusing {what} of stale run {d.name}: {reason}. These "
                f"artifacts were evaluated on the pre-correction "
                f"full-timeline cohort; rerun them under the corrected "
                f"train-only cohort (see RERUN.md)."
            )
        if require_predictions:
            current, reason = corrected_prediction_artifact(payload, metrics_path, verify_scaled_metrics=True)
            if not current:
                raise RuntimeError(
                    f"Refusing {what} of incomplete run {d.name}: {reason}."
                )
        validated.append(d)
    if not checked:
        raise RuntimeError(f"Refusing {what}: no runs with metrics.json under {runs_dir}")
    return validated


def prediction_truth_digest(payload: Mapping[str, Any], metrics_path: Path) -> str:
    """Fingerprint target identities and values for cross-run comparability.

    Shape-level validation cannot detect that one seed was evaluated against a
    different raw-data revision.  The workbook builder compares this digest
    across every run in an experiment before averaging any score.
    """
    prediction_name = payload.get("predictions_file")
    if prediction_name not in {"predictions.csv", "predictions_norm.csv"}:
        raise ValueError("missing normalized predictions provenance")
    path = Path(metrics_path).parent / prediction_name
    records = []
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not PREDICTION_COLUMNS.issubset(reader.fieldnames or []):
                raise ValueError(f"invalid prediction columns in {path.name}")
            for row in reader:
                identity = (
                    row["State"], row["Commodity"], row["Import/Export"],
                    date.fromisoformat(row["Time"][:10]).isoformat(),
                )
                targets = (
                    float(row["true_value"]), float(row["true_weight"]),
                    float(row["true_value_real"]), float(row["true_weight_real"]),
                )
                if not all(math.isfinite(value) for value in targets):
                    raise ValueError(f"non-finite target in {path.name}")
                records.append((*identity, *(value.hex() for value in targets)))
    except (OSError, csv.Error, UnicodeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot fingerprint {path}: {exc}") from exc
    canonical = "\n".join("\x1f".join(record) for record in sorted(records))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
