import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from src.inference.metrics import REQUIRED_EVALUATION_METRICS
from src.utils.provenance import (
    CORRECTED_TEST_TARGETS,
    cohort_provenance,
    corrected_aggregate_artifact,
    data_protocol,
    corrected_per_combo_artifact,
    corrected_prediction_artifact,
)
from src.utils.experiment_manifest import (
    exp1_expected_runs,
    exp1_protocol,
    exp1_source_domain,
    load_manifest,
)
from src.utils.run_contract import expected_exp1_run_contract


ROOT = Path(__file__).resolve().parents[1]


def _perfect_metrics():
    metrics = {name: 0.0 for name in REQUIRED_EVALUATION_METRICS}
    metrics.update({"r2": 1.0, "value_r2": 1.0, "weight_r2": 1.0})
    return metrics


def _normalized_payload_fields():
    return {
        "mode": "one_step_normalized",
        "prediction_units": "normalized",
        "scale_dependent_metric_units": "normalized",
        "percentage_metric_units": "real",
    }


def _run_contract_fields(
    *, model="lstm", variant="embeddings", seed=26, aggregate=False
):
    source_fingerprint = "a" * 64
    source_domain = {"states": ["A"]}
    identity = {
        "model": model,
        "variant": variant,
        "seed": seed,
        "aggregate": aggregate,
    }
    return {
        **identity,
        "source_fingerprint": source_fingerprint,
        "source_domain": source_domain,
        "run_contract": {
            "run_contract_version": 1,
            "benchmark_protocol_version": cohort_provenance(1, True)[
                "benchmark_protocol_version"
            ],
            "identity": identity,
            "model_config": {},
            "data_config": {},
            "training_config": {},
            "selection": {},
            "source_fingerprint": source_fingerprint,
            "source_domain": source_domain,
        },
    }


PREDICTION_HEADER = (
    "State,Commodity,Import/Export,Time,"
    "pred_value,pred_weight,true_value,true_weight,"
    "pred_value_real,pred_weight_real,true_value_real,true_weight_real\n"
)


def test_corrected_artifact_requires_train_only_918_cohort():
    payload = {
        **_normalized_payload_fields(),
        **_run_contract_fields(),
        "n_samples": CORRECTED_TEST_TARGETS,
        "predictions_file": "predictions.csv",
        "cohort": cohort_provenance(918, True),
        "data_protocol": data_protocol(36, 12, None),
        "metrics": _perfect_metrics(),
    }
    assert corrected_per_combo_artifact(payload) == (True, "current")

    without_contract = {**payload}
    without_contract.pop("run_contract")
    current, reason = corrected_per_combo_artifact(without_contract)
    assert current is False
    assert "run contract" in reason

    for bad in (
        {"n_samples": 4515},
        {**payload, "cohort": cohort_provenance(903, False)},
        {**payload, "n_samples": 4515},
        {**payload, "n_samples": "not-a-count"},
    ):
        assert corrected_per_combo_artifact(bad)[0] is False


def test_corrected_artifact_rejects_raw_or_missing_normalized_metric():
    base = {
        **_normalized_payload_fields(),
        **_run_contract_fields(),
        "n_samples": CORRECTED_TEST_TARGETS,
        "predictions_file": "predictions.csv",
        "cohort": cohort_provenance(918, True),
        "data_protocol": data_protocol(36, 12, None),
        "metrics": _perfect_metrics(),
    }
    assert corrected_per_combo_artifact({**base, "mode": "one_step"})[0] is False
    assert corrected_per_combo_artifact({
        **base, "mode": "one_step_normalized", "metrics": {"mae": 0.2},
    })[0] is False
    assert corrected_per_combo_artifact({
        **base, "prediction_units": "real",
    })[0] is False
    assert corrected_per_combo_artifact({
        **base, "scale_dependent_metric_units": "real",
    })[0] is False
    assert corrected_per_combo_artifact({
        **base, "percentage_metric_units": "normalized",
    })[0] is False
    partial = _perfect_metrics()
    partial.pop("smape")
    current, reason = corrected_per_combo_artifact({**base, "metrics": partial})
    assert current is False
    assert "smape" in reason


def test_corrected_aggregate_artifact_has_its_own_population_contract():
    source_fingerprint = "a" * 64
    source_domain = {"states": ["A"]}
    identity = {
        "model": "lstm", "variant": "embeddings", "seed": 26,
        "aggregate": True,
    }
    payload = {
        **identity,
        **_normalized_payload_fields(),
        "aggregate": True,
        "n_samples": 5,
        "predictions_file": "predictions.csv",
        "source_fingerprint": source_fingerprint,
        "source_domain": source_domain,
        "run_contract": {
            "run_contract_version": 1,
            "benchmark_protocol_version": cohort_provenance(1, True)[
                "benchmark_protocol_version"
            ],
            "identity": identity,
            "model_config": {},
            "data_config": {},
            "training_config": {},
            "selection": {},
            "source_fingerprint": source_fingerprint,
            "source_domain": source_domain,
        },
        "cohort": cohort_provenance(1, True),
        "data_protocol": data_protocol(36, 12, None),
        "metrics": _perfect_metrics(),
    }
    assert corrected_aggregate_artifact(payload) == (True, "current")
    assert corrected_aggregate_artifact({**payload, "n_samples": 4})[0] is False
    assert corrected_aggregate_artifact(
        {**payload, "cohort": cohort_provenance(918, True)}
    )[0] is False
    assert corrected_aggregate_artifact(
        {**payload, "cohort": cohort_provenance(1, False)}
    )[0] is False
    assert corrected_aggregate_artifact(
        {**payload, "data_protocol": {"input_len": 36}}
    )[0] is False


def test_prediction_artifact_requires_complete_matching_rows(tmp_path):
    metrics = tmp_path / "metrics.json"
    payload = {
        **_normalized_payload_fields(),
        "n_samples": 2,
        "predictions_file": "predictions.csv",
        "cohort": cohort_provenance(918, True),
        "data_protocol": data_protocol(36, 12, None),
        "metrics": _perfect_metrics(),
    }
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        PREDICTION_HEADER
        + "A,C,Import,2025-01-01,0,0,0,0,10,20,10,20\n"
    )
    current, reason = corrected_prediction_artifact(payload, metrics)
    assert current is False
    assert "has 1 rows; expected 2" in reason


def test_prediction_artifact_binds_every_direct_metric_to_real_columns(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    payload = {
        **_normalized_payload_fields(),
        "n_samples": 2,
        "predictions_file": "predictions.csv",
        "metrics": _perfect_metrics(),
    }
    (tmp_path / "predictions.csv").write_text(
        PREDICTION_HEADER
        + "A,C,Import,2025-01-01,0.1,0.2,0.1,0.2,101,202,101,202\n"
        + "A,C,Import,2025-02-01,0.2,0.4,0.2,0.4,102,204,102,204\n"
    )
    expected = {
        "expected_num_combos": 1,
        "expected_months": 2,
        "expected_dates": {date(2025, 1, 1), date(2025, 2, 1)},
        "expected_combos": {("A", "C", "Import")},
    }
    assert corrected_prediction_artifact(
        payload, metrics_path, **expected
    ) == (True, "current")

    tampered = {**payload, "metrics": {**payload["metrics"], "smape": 1.0}}
    current, reason = corrected_prediction_artifact(
        tampered, metrics_path, **expected
    )
    assert current is False
    assert "metric smape" in reason

    wrong_units = {**payload, "prediction_units": "real"}
    current, reason = corrected_prediction_artifact(
        wrong_units, metrics_path, **expected
    )
    assert current is False
    assert "not declared normalized" in reason


def test_run_is_current_rejects_legacy_metrics(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"n_samples": 4515, "metrics": {}}))
    result = subprocess.run(
        [sys.executable, "scripts/run_is_current.py", str(metrics)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "stale" in result.stdout


def test_run_is_current_requires_complete_declared_predictions(tmp_path):
    metrics = tmp_path / "metrics.json"
    payload = {
        **_run_contract_fields(),
        **_normalized_payload_fields(),
        "n_samples": CORRECTED_TEST_TARGETS,
        "predictions_file": "predictions.csv",
        "cohort": cohort_provenance(918, True),
        "data_protocol": data_protocol(36, 12, None),
        "metrics": _perfect_metrics(),
    }
    metrics.write_text(json.dumps(payload))
    result = subprocess.run(
        [sys.executable, "scripts/run_is_current.py", str(metrics)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "missing predictions artifact" in result.stdout

    rows = []
    for combo in range(918):
        for month in range(1, 6):
            value_norm = month / 10
            weight_norm = month / 20
            value_real = 100 + month
            weight_real = 200 + 2 * month
            rows.append(
                f"S{combo},C,Import,2025-{month:02d}-01,"
                f"{value_norm},{weight_norm},{value_norm},{weight_norm},"
                f"{value_real},{weight_real},{value_real},{weight_real}\n"
            )
    (tmp_path / "predictions.csv").write_text(PREDICTION_HEADER + "".join(rows))
    result = subprocess.run(
        [sys.executable, "scripts/run_is_current.py", str(metrics)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0

    matching = subprocess.run(
        [
            sys.executable, "scripts/run_is_current.py", str(metrics),
            "--expected-model", "lstm", "--expected-variant", "embeddings",
            "--expected-seed", "26",
        ],
        capture_output=True, text=True,
    )
    assert matching.returncode == 0

    wrong_identity = subprocess.run(
        [
            sys.executable, "scripts/run_is_current.py", str(metrics),
            "--expected-model", "gru", "--expected-variant", "embeddings",
            "--expected-seed", "26",
        ],
        capture_output=True, text=True,
    )
    assert wrong_identity.returncode == 1
    assert "model='lstm', expected 'gru'" in wrong_identity.stdout

    payload["data_protocol"]["train_target_start"] = "2015-01-01"
    metrics.write_text(json.dumps(payload))
    explicit_null = subprocess.run(
        [
            sys.executable, "scripts/run_is_current.py", str(metrics),
            "--train-target-start", "null",
        ],
        capture_output=True, text=True,
    )
    assert explicit_null.returncode == 1
    assert "train_target_start='2015-01-01', expected None" in explicit_null.stdout


def test_exp1_manifest_skip_gate_rejects_off_protocol_training_contract(tmp_path):
    manifest_path = ROOT / "config" / "experiment_manifest.env"
    manifest = load_manifest(manifest_path)
    protocol = exp1_protocol(manifest)
    source_domain = exp1_source_domain(manifest)
    run = next(
        item for item in exp1_expected_runs(manifest)
        if (item.model, item.variant, item.seed) == ("lstm", "embeddings", 26)
    )
    contract = expected_exp1_run_contract(ROOT, run, protocol, source_domain)
    payload = {
        "model": run.model,
        "variant": run.variant,
        "seed": run.seed,
        "aggregate": True,
        **_normalized_payload_fields(),
        "n_samples": 5,
        "predictions_file": "predictions.csv",
        "source_fingerprint": protocol["source_fingerprint"],
        "source_domain": source_domain,
        "run_contract": contract,
        "cohort": cohort_provenance(1, True),
        "data_protocol": data_protocol(36, 12, None),
        "metrics": _perfect_metrics(),
    }
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(payload))
    from src.utils.exp1_truth import load_exp1_truth
    import torch
    from src.utils.training_completion import write_completion
    checkpoint = tmp_path / "best.pth"
    torch.save({"run_contract": contract, "epoch": 0, "score": 0.0}, checkpoint)
    patience = contract["training_config"]["patience"]
    payload["training_completion"] = write_completion(
        checkpoint, contract, epochs_run=patience + 1, best_epoch=1,
        best_val=0.0, stopped_early=True, patience_counter=patience,
    )
    metrics.write_text(json.dumps(payload))
    truth = load_exp1_truth(ROOT)
    predictions = PREDICTION_HEADER + "".join(
        ",".join([*key, *(str(x) for x in [*v[:2], *v[:2], *v[2:], *v[2:]])]) + "\n"
        for key, v in truth["targets"].items()
    )
    (tmp_path / "predictions.csv").write_text(predictions)
    command = [
        sys.executable, "scripts/run_is_current.py", str(metrics),
        "--aggregate", "--exp1-manifest", str(manifest_path),
        "--input-len", "36", "--lag-count", "12",
        "--train-target-start", "null", "--expected-seed", "26",
        "--expected-model", "lstm", "--expected-variant", "embeddings",
    ]
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0

    completion = payload.pop("training_completion")
    metrics.write_text(json.dumps(payload))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "completion evidence differs" in result.stdout
    payload["training_completion"] = completion
    metrics.write_text(json.dumps(payload))

    assert payload["run_contract"]["numerics"] == {"allow_tf32": False}
    for numerics in (None, {"allow_tf32": True}):
        if numerics is None:
            payload["run_contract"].pop("numerics")
        else:
            payload["run_contract"]["numerics"] = numerics
        metrics.write_text(json.dumps(payload))
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 1
        assert "does not match the canonical Exp 1" in result.stdout
    payload["run_contract"]["numerics"] = {"allow_tf32": False}
    metrics.write_text(json.dumps(payload))

    # Invent different targets and identical predictions. All direct metrics
    # remain perfect, but the authoritative source check must still reject.
    forged = PREDICTION_HEADER + "".join(
        ",".join([*key, *(str(x) for x in [
            *(v[:2] + 0.1), *(v[:2] + 0.1),
            *(v[2:] + truth["scale"] * 0.1), *(v[2:] + truth["scale"] * 0.1),
        ])]) + "\n" for key, v in truth["targets"].items()
    )
    (tmp_path / "predictions.csv").write_text(forged)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "test targets differ from pinned source" in result.stdout
    (tmp_path / "predictions.csv").write_text(predictions)

    payload["run_contract"]["training_config"]["lr"] = 0.5
    metrics.write_text(json.dumps(payload))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "does not match the canonical Exp 1" in result.stdout
