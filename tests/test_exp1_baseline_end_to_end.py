"""Real-source CPU baseline -> canonical Exp1 artifact validation."""
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_implicit_and_explicit_target_start_have_the_same_contract():
    from src.utils.run_contract import semantic_data_config
    assert semantic_data_config({"data": {}}) == semantic_data_config(
        {"data": {"train_target_start": None}}
    )
    assert semantic_data_config({"data": {"train_target_start": "2015-01-01"}}) != (
        semantic_data_config({"data": {}})
    )


@pytest.mark.parametrize("model", ["random_walk", "moving_average", "seasonal_naive"])
def test_baseline_defaults_preserve_exp1_recipe_and_validate(tmp_path, model):
    out = tmp_path / model
    env = {**os.environ, "DATA_DIR": str(ROOT)}
    completed = subprocess.run([
        sys.executable, "scripts/run_baseline.py", "--model", model,
        "--aggregate", "--seed", "26", "--normalized", "--device", "cpu",
        "--out-dir", str(out),
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads((out / "metrics.json").read_text())
    definition = json.loads((out / "baseline_definition.json").read_text())
    inventory = json.loads((out / "checkpoint_manifest.json").read_text())
    assert definition["kind"] == inventory["kind"] == "parameter_free_baseline"
    assert definition["model_config"]["name"] == model
    assert definition["run_contract"] == inventory["run_contract"] == payload["run_contract"]
    assert payload["run_contract"]["training_config"]["batch_size"] == 33
    assert payload["n_samples"] == 5
    validated = subprocess.run([
        sys.executable, "scripts/run_is_current.py", str(out / "metrics.json"),
        "--aggregate", "--exp1-manifest", "config/experiment_manifest.env",
        "--expected-model", model, "--expected-variant", "none",
        "--expected-seed", "26", "--input-len", "36", "--lag-count", "12",
        "--train-target-start", "null",
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert validated.returncode == 0, validated.stdout + validated.stderr
