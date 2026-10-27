"""Lightning completion adapter; importable by either repository namespace."""
import json
import os
from pathlib import Path

try:
    from .training_completion import checkpoint_digest, completion_path, validate_completion
except ImportError:  # external/s4 owns the top-level src package
    from training_completion import checkpoint_digest, completion_path, validate_completion


def validate_s4_completion(receipt, contract, checkpoint, *, require_early_stopping=True):
    if not isinstance(receipt, dict) or receipt.get("kind") != "upstream_s4":
        raise ValueError("missing S4 training completion evidence")
    if receipt.get("run_contract") != contract:
        raise ValueError("S4 completion belongs to a different run contract")
    training = contract["training_config"]
    policy = {**contract, "training_config": {
        "epochs": training["trainer"]["max_epochs"],
        "patience": training["callbacks"]["early_stopping"]["patience"],
    }}
    validate_completion({**receipt, "run_contract": policy}, policy, checkpoint,
                        require_early_stopping=require_early_stopping)


def read_s4_completion(checkpoint, contract, *, require_early_stopping=True):
    path = completion_path(checkpoint)
    if not path.is_file():
        raise ValueError(f"No S4 training completion evidence at {path}; retrain")
    receipt = json.loads(path.read_text())
    validate_s4_completion(receipt, contract, checkpoint,
                           require_early_stopping=require_early_stopping)
    return receipt


def write_s4_completion(checkpoint, contract, *, epochs_run, best_val,
                        stopped_early, patience_counter):
    import torch
    selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if selected.get("run_contract") != contract:
        raise ValueError("selected S4 checkpoint differs from completed training contract")
    selections = [state for state in selected.get("callbacks", {}).values()
                  if isinstance(state, dict) and "best_model_score" in state]
    if (len(selections) != 1 or selections[0]["best_model_score"] is None
            or float(selections[0]["best_model_score"]) != float(best_val)):
        raise ValueError("selected S4 checkpoint validation score differs from completed training")
    receipt = {"version": 1, "kind": "upstream_s4", "state": "finished",
               "stop_reason": "early_stopping" if stopped_early else "epoch_limit",
               "epochs_run": epochs_run, "best_epoch": selected["epoch"] + 1,
               "best_val": float(best_val), "patience_counter": patience_counter,
               "run_contract": contract, "checkpoint_sha256": checkpoint_digest(checkpoint)}
    validate_s4_completion(receipt, contract, checkpoint, require_early_stopping=False)
    path = completion_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt
