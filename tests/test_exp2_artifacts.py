"""The common publication loader must validate the complete paired roster."""
import json
from pathlib import Path

import pytest

from src.utils import exp2_artifacts as artifacts
from src.utils.exp2_protocol import expected_runs
from src.utils.experiment_manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def roster(tmp_path, monkeypatch):
    runs = expected_runs(load_manifest(ROOT / "config/experiment_manifest.env"))
    for run in runs:
        folder = tmp_path / run.directory_name
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps({
            "model": run.model, "variant": run.variant, "seed": run.seed,
        }))
    calls = []
    monkeypatch.setattr(artifacts, "corrected_per_combo_artifact", lambda p: (True, "ok"))
    monkeypatch.setattr(artifacts, "validate_contract", lambda p, repo: (True, "ok"))
    monkeypatch.setattr(artifacts, "validate_exp2_completion_artifact",
                        lambda p, path: calls.append(("completion", path)))
    def check_predictions(payload, path, *, verify_scaled_metrics):
        assert verify_scaled_metrics is True
        calls.append(("source_predictions", path))
        return True, "ok"
    monkeypatch.setattr(artifacts, "corrected_prediction_artifact", check_predictions)
    monkeypatch.setattr(artifacts, "prediction_truth_digest", lambda p, path: "same_source_truth")
    return tmp_path, runs, calls


def test_all_consumed_runs_receive_source_and_completion_checks(roster):
    directory, runs, calls = roster
    result = artifacts.validated_exp2_artifacts(ROOT, directory)
    assert len(result) == len(runs) == 171
    assert {r for r, _, _ in result} == set(runs)
    assert len(calls) == 2 * len(runs)
    assert {path.parent.name for kind, path in calls if kind == "completion"} == {
        r.directory_name for r in runs
    }


@pytest.mark.parametrize("change", ["missing_seed", "extra_arm"])
def test_partial_or_different_comparison_families_fail_before_analysis(roster, change):
    directory, runs, calls = roster
    if change == "missing_seed":
        (directory / runs[0].directory_name / "metrics.json").unlink()
    else:
        folder = directory / "unregistered_none_seed26"
        folder.mkdir()
        (folder / "metrics.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Inexact Exp 2 roster/manifest"):
        artifacts.validated_exp2_artifacts(ROOT, directory)
    assert calls == []


@pytest.mark.parametrize("field,value", [("seed", 105), ("seed", True),
                                         ("model", "random_walk"), ("variant", "none")])
def test_directory_identity_cannot_relabel_payload(roster, field, value):
    directory, runs, _ = roster
    path = directory / runs[0].directory_name / "metrics.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="directory/identity mismatch"):
        artifacts.validated_exp2_artifacts(ROOT, directory)


@pytest.mark.parametrize("check", ["recipe", "completion", "predictions", "truth"])
def test_no_artifact_list_is_returned_when_any_validation_fails(roster, monkeypatch, check):
    directory, runs, _ = roster
    if check == "recipe":
        monkeypatch.setattr(artifacts, "validate_contract", lambda *args: (False, "wrong recipe"))
    elif check == "completion":
        def reject(*args):
            raise ValueError("missing completion")
        monkeypatch.setattr(artifacts, "validate_exp2_completion_artifact", reject)
    elif check == "predictions":
        monkeypatch.setattr(artifacts, "corrected_prediction_artifact",
                            lambda *a, **kw: (False, "wrong source predictions"))
    else:
        monkeypatch.setattr(artifacts, "prediction_truth_digest", lambda p, path: path.parent.name)
    with pytest.raises(RuntimeError, match="Refusing to publish"):
        artifacts.validated_exp2_artifacts(ROOT, directory)


def test_paper_builder_uses_shared_loader(monkeypatch):
    from scripts.build_paper_results import _exp2_rows
    def reject(*args, **kwargs):
        raise RuntimeError("shared validator rejected artifacts")
    monkeypatch.setattr(artifacts, "validated_exp2_artifacts", reject)
    with pytest.raises(RuntimeError, match="shared validator rejected"):
        _exp2_rows(ROOT)
