"""Exp1 terminal training evidence, bound to the selected model bytes."""
import json
from pathlib import Path

from .training_completion import checkpoint_digest, clean_contract, read_completion

BASELINES = {"seasonal_naive", "random_walk", "moving_average"}
TREES = {"xgboost", "lightgbm", "random_forest"}


def read_exp1_completion(checkpoint, contract, *, require_early_stopping=True):
    checkpoint = Path(checkpoint)
    model = contract["identity"]["model"]
    if model not in TREES:
        return read_completion(checkpoint, contract,
                               require_early_stopping=require_early_stopping)
    # Tree fit publishes metadata last, only after both fitted estimators were
    # saved. Verify without deserializing executable joblib content in a gate.
    metadata = json.loads(Path(f"{checkpoint}_metadata.json").read_text())
    if (not isinstance(metadata, dict)
            or type(metadata.get("artifact_schema_version")) is not int
            or metadata.get("artifact_schema_version") != 1
            or metadata.get("run_contract") != clean_contract(contract)):
        raise ValueError("native completion has a missing or mismatched run contract")
    hashes = metadata.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("native completion has no component hashes")
    for name in ("value", "weight"):
        path = Path(f"{checkpoint}_{name}.joblib")
        actual = {"bytes": path.stat().st_size, "sha256": checkpoint_digest(path)}
        if hashes.get(name) != actual:
            raise ValueError(f"native completion checkpoint differs: {name}")
    return {"version": 1, "kind": "native_tree", "state": "finished",
            "run_contract": clean_contract(contract), "artifact_hashes": hashes}


def validate_exp1_completion_artifact(payload, metrics_path):
    model = payload["model"]
    if model in BASELINES:
        return
    checkpoint = Path(metrics_path).parent / ("best_xgb" if model in TREES else "best.pth")
    receipt = read_exp1_completion(checkpoint, payload["run_contract"])
    if receipt != payload.get("training_completion"):
        raise ValueError("Exp1 metrics completion evidence differs from selected checkpoint")
