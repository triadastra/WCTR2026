import json

import pytest

from src.utils.provenance import BENCHMARK_PROTOCOL_VERSION

from src.inference.cohort import (
    apply_checkpoint_cohort_config,
    validate_checkpoint_cohort,
)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
          "num_combos": 918, "combos": [["A", "B", "Import"]],
          "cohort_from_train_only": True}, True),
        ({"benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
          "num_combos": 903, "combos": [["A", "B", "Import"]]}, False),
    ],
)
def test_checkpoint_artifact_controls_inference_cohort(tmp_path, metadata, expected):
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    checkpoint = run_dir / "best.pth"
    checkpoint.touch()
    (artifacts / "kept_combos.json").write_text(json.dumps(metadata))
    cfg = {"data": {"cohort_from_train_only": not expected}}

    loaded = apply_checkpoint_cohort_config(cfg, checkpoint)

    assert cfg["data"]["cohort_from_train_only"] is expected
    assert loaded["num_combos"] == metadata["num_combos"]


def test_checkpoint_cohort_validation_rejects_different_series(tmp_path):
    artifact_path = tmp_path / "kept_combos.json"
    metadata = {
        "_artifact_path": str(artifact_path),
        "num_combos": 1,
        "combos": [["A", "B", "Import"]],
    }

    with pytest.raises(ValueError, match="does not match the checkpoint artifact"):
        validate_checkpoint_cohort([("A", "C", "Import")], metadata)


def test_checkpoint_artifact_rejects_pre_fairness_protocol(tmp_path):
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    checkpoint = run_dir / "best.pth"
    checkpoint.touch()
    (artifacts / "kept_combos.json").write_text(json.dumps({
        "num_combos": 918,
        "cohort_from_train_only": True,
    }))
    with pytest.raises(ValueError, match="retraining is required"):
        apply_checkpoint_cohort_config({"data": {}}, checkpoint)
