import copy
import pytest
import torch

from src.utils.exp2_completion import validate_exp2_completion_artifact
from src.utils.s4_completion import write_s4_completion, read_s4_completion


def s4_artifact(tmp_path, *, stopped=True):
    contract = {"identity": {"model": "s4", "variant": "embeddings", "seed": 26},
                "training_config": {"loss_reduction": "valid-target-mean-v1",
                                    "trainer": {"max_epochs": 5},
                                    "callbacks": {"early_stopping": {"patience": 2}}}}
    path = tmp_path / "hydra/checkpoints/val/loss.ckpt"
    path.parent.mkdir(parents=True)
    torch.save({"epoch": 0, "state_dict": {}, "run_contract": contract,
                "callbacks": {"ModelCheckpoint": {"best_model_score": torch.tensor(1.0)}}}, path)
    receipt = write_s4_completion(path, contract, epochs_run=3 if stopped else 5,
                                  best_val=1.0, stopped_early=stopped,
                                  patience_counter=2 if stopped else 1)
    return path, {"model": "s4", "run_contract": contract, "training_completion": receipt}


def test_upstream_completion_binds_metrics_to_selected_checkpoint(tmp_path):
    path, payload = s4_artifact(tmp_path)
    validate_exp2_completion_artifact(payload, tmp_path / "metrics.json")
    payload["training_completion"] = None
    with pytest.raises(ValueError, match="evidence differs"):
        validate_exp2_completion_artifact(payload, tmp_path / "metrics.json")
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checkpoint differs"):
        read_s4_completion(path, payload["run_contract"])


def test_upstream_epoch_cap_is_not_early_stopping(tmp_path):
    path, payload = s4_artifact(tmp_path, stopped=False)
    assert read_s4_completion(path, payload["run_contract"], require_early_stopping=False)
    with pytest.raises(ValueError, match="epoch-capped"):
        validate_exp2_completion_artifact(payload, tmp_path / "metrics.json")


def test_upstream_incomplete_fit_cannot_publish_receipt(tmp_path):
    path, payload = s4_artifact(tmp_path)
    with pytest.raises(ValueError, match="configured epoch limit"):
        write_s4_completion(path, payload["run_contract"], epochs_run=2, best_val=1.0,
                            stopped_early=False, patience_counter=1)


def test_upstream_contract_mismatch_is_rejected(tmp_path):
    path, payload = s4_artifact(tmp_path)
    wrong = copy.deepcopy(payload["run_contract"])
    wrong["identity"]["seed"] = 105
    with pytest.raises(ValueError, match="different run contract"):
        read_s4_completion(path, wrong)


def test_exp2_root_neural_requires_completion(tmp_path):
    payload = {"model": "lstm", "run_contract": {
        "identity": {"model": "lstm"}, "training_config": {}}}
    with pytest.raises(ValueError, match="No training completion"):
        validate_exp2_completion_artifact(payload, tmp_path / "metrics.json")


def test_exp2_baseline_requires_no_training_receipt(tmp_path):
    validate_exp2_completion_artifact({"model": "random_walk"}, tmp_path / "metrics.json")


@pytest.mark.parametrize("score", [2.0, float("nan"), None])
def test_upstream_selected_score_must_match_terminal_summary(tmp_path, score):
    path, payload = s4_artifact(tmp_path)
    selected = torch.load(path, weights_only=False)
    selected["callbacks"]["ModelCheckpoint"]["best_model_score"] = score
    torch.save(selected, path)
    with pytest.raises(ValueError, match="validation score differs"):
        write_s4_completion(path, payload["run_contract"], epochs_run=3, best_val=1.0,
                            stopped_early=True, patience_counter=2)
