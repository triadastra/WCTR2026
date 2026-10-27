"""Real-source regression coverage for publication metrics and seed identity."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import pandas as pd

from src.inference.metrics import SCALED_METRIC_NAMES, compute_evaluation_metrics
from src.utils.provenance import corrected_prediction_artifact

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("review_baseline")
    directory = root / "random_walk_none_seed26"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_baseline.py"),
        "--model", "random_walk", "--seed", "26", "--normalized",
        "--device", "cpu", "--data-dir", str(ROOT), "--out-dir", str(directory),
    ], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    path = directory / "metrics.json"
    return path, json.loads(path.read_text())


def test_real_source_scaled_metrics_pass(baseline, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(ROOT))
    path, payload = baseline
    assert corrected_prediction_artifact(payload, path, verify_scaled_metrics=True) == (True, "current")


@pytest.mark.parametrize("metric", SCALED_METRIC_NAMES)
def test_each_fabricated_scaled_metric_is_rejected(baseline, monkeypatch, metric):
    monkeypatch.setenv("DATA_DIR", str(ROOT))
    path, original = baseline
    payload = copy.deepcopy(original)
    payload["metrics"][metric] = 123456.0
    valid, reason = corrected_prediction_artifact(payload, path, verify_scaled_metrics=True)
    assert not valid
    assert metric in reason and "seasonal scaling" in reason


def test_exp2_skip_gate_rejects_changed_scaled_metrics(baseline):
    path, original = baseline
    changed = copy.deepcopy(original)
    changed["metrics"]["mase"] = 123456.0
    try:
        path.write_text(json.dumps(changed))
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/run_is_current.py"), str(path),
            "--exp2", "--expected-model", "random_walk", "--expected-variant", "none",
            "--expected-seed", "26",
        ], cwd=ROOT, env={**os.environ, "DATA_DIR": str(ROOT)},
            capture_output=True, text=True, timeout=180)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "mase differs from pinned source seasonal scaling" in result.stdout
    finally:
        path.write_text(json.dumps(original))


def test_collector_rejects_renamed_seed_without_overwriting_results(baseline, tmp_path):
    _, payload = baseline
    directory = tmp_path / "random_walk_none_seed105"
    directory.mkdir()
    (directory / "metrics.json").write_text(json.dumps(payload))
    output = tmp_path / "results.csv"
    output.write_text("previous results\n")
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/collect_results.py"), "--runs-dir", str(tmp_path),
    ], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "directory/payload seed mismatch" in result.stderr
    assert output.read_text() == "previous results\n"


@pytest.mark.parametrize("mutation,reason", [
    ("invented_truth", "test targets differ from pinned source"),
    ("real_truth", "target inverse transforms differ"),
    ("real_predictions", "prediction inverse transforms differ"),
    ("identity", "target identities differ"),
])
def test_self_consistent_prediction_files_are_bound_to_source(baseline, tmp_path, mutation, reason):
    path, original = baseline
    payload = copy.deepcopy(original)
    frame = pd.read_csv(path.parent / payload["predictions_file"])
    if mutation == "invented_truth":
        for pred, target in (("pred_value", "true_value"), ("pred_weight", "true_weight"),
                             ("pred_value_real", "true_value_real"), ("pred_weight_real", "true_weight_real")):
            frame[target] = 2 * frame[target] + 10
            frame[pred] = frame[target]
        payload["metrics"].update({name: 0.0 for name in SCALED_METRIC_NAMES})
    elif mutation == "identity":
        frame.loc[frame.State == frame.State.iloc[0], "State"] = "invented_state"
    elif mutation == "real_truth":
        frame[["true_value_real", "true_weight_real"]] *= 2
    else:
        frame[["pred_value_real", "pred_weight_real"]] *= 2
    columns = [("pred_value", "pred_weight"), ("true_value", "true_weight"),
               ("pred_value_real", "pred_weight_real"), ("true_value_real", "true_weight_real")]
    payload["metrics"].update(compute_evaluation_metrics(
        *(frame[list(c)].to_numpy() for c in columns)
    ))
    frame.to_csv(tmp_path / payload["predictions_file"], index=False)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(payload))
    valid, message = corrected_prediction_artifact(payload, metrics_path, verify_scaled_metrics=True)
    assert not valid
    assert reason in message


def test_source_validation_aligns_by_identity_not_csv_order(baseline, tmp_path):
    path, payload = baseline
    frame = pd.read_csv(path.parent / payload["predictions_file"])
    frame.sample(frac=1, random_state=7).to_csv(tmp_path / payload["predictions_file"], index=False)
    assert corrected_prediction_artifact(
        payload, tmp_path / "metrics.json", verify_scaled_metrics=True
    ) == (True, "current")
