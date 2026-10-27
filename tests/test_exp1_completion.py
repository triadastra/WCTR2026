from copy import deepcopy

import pytest

from src.models.tabular_common import save_tree_bundle
from src.utils.exp1_completion import read_exp1_completion, validate_exp1_completion_artifact
from tests.test_exp3_completion import trainer, fit


def test_exp1_neural_requires_terminal_selected_checkpoint(tmp_path):
    t, contract = trainer(tmp_path)
    contract["identity"] = {"model": "lstm"}
    fit(t)
    checkpoint = tmp_path / "checkpoints/best.pth"
    receipt = read_exp1_completion(checkpoint, contract)
    payload = {"model": "lstm", "run_contract": contract, "training_completion": receipt}
    validate_exp1_completion_artifact(payload, checkpoint.parent / "metrics.json")
    payload["training_completion"] = None
    with pytest.raises(ValueError, match="completion evidence differs"):
        validate_exp1_completion_artifact(payload, checkpoint.parent / "metrics.json")


def test_exp1_completed_smoke_can_be_inspected_but_not_published(tmp_path):
    t, contract = trainer(tmp_path, epochs=1)
    contract["identity"] = {"model": "lstm"}
    fit(t)
    checkpoint = tmp_path / "checkpoints/best.pth"
    receipt = read_exp1_completion(checkpoint, contract, require_early_stopping=False)
    assert receipt["stop_reason"] == "epoch_limit"
    with pytest.raises(ValueError, match="epoch-capped"):
        read_exp1_completion(checkpoint, contract)


def test_exp1_native_completion_is_bound_to_both_estimators(tmp_path):
    contract = {"identity": {"model": "xgboost"}, "training_config": {}}
    checkpoint = tmp_path / "best_xgb"
    save_tree_bundle(checkpoint, {"fitted": "value"}, {"fitted": "weight"},
                     {"run_contract": contract})
    receipt = read_exp1_completion(checkpoint, contract)
    payload = {"model": "xgboost", "run_contract": contract, "training_completion": receipt}
    validate_exp1_completion_artifact(payload, tmp_path / "metrics.json")
    changed = deepcopy(contract)
    changed["training_config"]["lr"] = 0.5
    with pytest.raises(ValueError, match="mismatched run contract"):
        read_exp1_completion(checkpoint, changed)
    (tmp_path / "best_xgb_weight.joblib").write_bytes(b"changed estimator")
    with pytest.raises(ValueError, match="checkpoint differs"):
        validate_exp1_completion_artifact(payload, tmp_path / "metrics.json")


def test_exp1_baseline_needs_no_training_checkpoint(tmp_path):
    validate_exp1_completion_artifact({"model": "seasonal_naive"}, tmp_path / "metrics.json")


def test_exp1_collector_checks_completion_before_publishing(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import build_paper_results as collector
    run = SimpleNamespace(model="lstm", variant="embeddings", seed=26,
                          kind="neural", directory_name="lstm_embeddings_seed26")
    directory = tmp_path / "outputs/agg" / run.directory_name
    directory.mkdir(parents=True)
    payload = {"model": "lstm", "variant": "embeddings", "seed": 26,
               "aggregate": True, "run_contract": {"identity": {"model": "lstm"}}}
    (directory / "metrics.json").write_text(json.dumps(payload))
    # Isolate the collector's completion enforcement point. Source/recipe
    # gate behavior is exercised separately by the canonical CLI gate tests.
    monkeypatch.setattr(collector, "load_manifest", lambda _: {})
    monkeypatch.setattr(collector, "exp1_expected_runs", lambda _: [run])
    monkeypatch.setattr(collector, "exp1_protocol", lambda _: {})
    monkeypatch.setattr(collector, "corrected_aggregate_artifact", lambda _: (True, "ok"))
    monkeypatch.setattr(collector, "_validate_exp1_contract", lambda *args: None)
    with pytest.raises(RuntimeError, match="No training completion evidence"):
        collector._exp1_rows(tmp_path)


@pytest.mark.parametrize("metadata", [None, [], True, {"artifact_schema_version": True}])
def test_malformed_native_completion_fails_explicitly(tmp_path, metadata):
    import json
    (tmp_path / "best_xgb_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="native completion"):
        read_exp1_completion(tmp_path / "best_xgb", {"identity": {"model": "xgboost"}})


@pytest.mark.parametrize("field,value", [
    ("version", True), ("checkpoint_sha256", "z" * 64),
    ("epochs_run", True), ("best_epoch", 1.0),
])
def test_neural_completion_schema_is_strict(field, value):
    from src.utils.training_completion import validate_completion
    contract = {"training_config": {"epochs": 10, "patience": 2}}
    receipt = {"version": 1, "state": "finished", "stop_reason": "early_stopping",
               "run_contract": contract, "epochs_run": 3, "best_epoch": 1,
               "patience_counter": 2, "best_val": 0.0, "checkpoint_sha256": "a" * 64}
    validate_completion(receipt, contract)
    receipt[field] = value
    with pytest.raises(ValueError):
        validate_completion(receipt, contract)


@pytest.mark.parametrize("training", [None, [], {"epochs": True, "patience": 2},
                                      {"epochs": 10, "patience": "2"}])
def test_neural_completion_rejects_malformed_training_configuration(training):
    from src.utils.training_completion import validate_completion
    contract = {"training_config": training}
    receipt = {"version": 1, "state": "finished", "stop_reason": "early_stopping",
               "run_contract": contract}
    with pytest.raises(ValueError, match="invalid completion"):
        validate_completion(receipt, contract)
