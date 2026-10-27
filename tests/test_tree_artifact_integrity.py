from pathlib import Path
import json

import pytest

from src.models.tabular_common import load_tree_bundle, save_tree_bundle


def test_tree_bundle_hash_binds_both_estimators(tmp_path: Path):
    prefix = tmp_path / "best_xgb"
    contract = {"run_contract_version": 1, "identity": {"model": "test"}}
    save_tree_bundle(prefix, {"tree": "value"}, {"tree": "weight"}, {
        "variant": "onehot",
        "run_contract": contract,
    })

    value, weight, metadata = load_tree_bundle(prefix)
    assert value == {"tree": "value"}
    assert weight == {"tree": "weight"}
    assert metadata["run_contract"] == contract
    inventory = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert inventory["run_contract"] == contract
    assert {entry["role"] for entry in inventory["files"]} == {
        "value_estimator", "weight_estimator", "metadata"}

    value_path = Path(f"{prefix}_value.joblib")
    damaged = bytearray(value_path.read_bytes())
    damaged[-1] ^= 1
    value_path.write_bytes(damaged)
    with pytest.raises(ValueError, match="integrity validation"):
        load_tree_bundle(prefix)


def test_tree_bundle_rejects_legacy_unhashed_metadata(tmp_path: Path):
    prefix = tmp_path / "legacy"
    Path(f"{prefix}_metadata.json").write_text('{"variant": "onehot"}')
    with pytest.raises(ValueError, match="legacy"):
        load_tree_bundle(prefix)
