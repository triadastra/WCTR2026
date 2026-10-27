"""Real CLI round trips: CPU models plus CUDA-only Mamba when available."""
import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from src.utils.exp1_truth import load_exp1_truth, validate_exp1_truth
from src.utils.provenance import corrected_aggregate_artifact, corrected_prediction_artifact

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model", ["lstm", "gru", "transformer", "gpt", "s4", "s4_bidir", "xgboost", "lightgbm", "mamba"])
@pytest.mark.parametrize("variant", ["embeddings", "onehot"])
def test_learned_checkpoint_roundtrip_on_original_exp1_data(tmp_path, model, variant):
    import torch
    if model == "mamba" and not torch.cuda.is_available():
        pytest.skip("Real Mamba numerical/CLI verification requires CUDA and Triton")
    device = "cuda" if model == "mamba" else "cpu"
    cfg = yaml.safe_load((ROOT / "config/base.yaml").read_text())
    cfg["training"].update(device=device, epochs=1)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / model
    checkpoint = out / ("best_xgb" if model in {"xgboost", "lightgbm"} else "best.pth")
    env = {**os.environ, "DATA_DIR": str(ROOT)}
    common = ["--model", model, "--variant", variant, "--aggregate",
              "--seed", "26", "--config", str(config_path), "--out-dir", str(out)]
    for script, extra in [
        ("scripts/train.py", []),
        ("scripts/inference.py", ["--checkpoint", str(checkpoint),
                                   "--normalized", "--device", device]),
    ]:
        result = subprocess.run([sys.executable, script, *common, *extra],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    metrics_path = out / "metrics.json"
    payload = json.loads(metrics_path.read_text())
    inventory = json.loads((out / "checkpoint_manifest.json").read_text())
    assert inventory["run_contract"] == payload["run_contract"]
    for entry in inventory["files"]:
        artifact = (out / entry["path"]).read_bytes()
        assert len(artifact) == entry["bytes"]
        assert hashlib.sha256(artifact).hexdigest() == entry["sha256"]
    assert corrected_aggregate_artifact(payload)[0]
    assert corrected_prediction_artifact(
        payload, metrics_path, expected_num_combos=1,
        expected_combos={("ALL", "ALL", "ALL")},
    )[0]
    assert validate_exp1_truth(payload, metrics_path, load_exp1_truth(ROOT))[0]
    if model in {"xgboost", "lightgbm"}:
        assert (out / "best_xgb_metadata.json").is_file()
        assert (out / "best_xgb_value.joblib").is_file()
        assert (out / "best_xgb_weight.joblib").is_file()
    else:
        assert (out / "last.pth").is_file()
    # A valid smoke artifact must not masquerade as the full paper recipe.
    result = subprocess.run([
        sys.executable, "scripts/run_is_current.py", str(metrics_path), "--aggregate",
        "--exp1-manifest", "config/experiment_manifest.env", "--expected-model", model,
        "--expected-variant", variant, "--expected-seed", "26",
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "canonical Exp 1" in result.stdout
