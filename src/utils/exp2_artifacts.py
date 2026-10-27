"""One complete, source-checked Exp 2 roster for tables and statistical figures."""
import json
from pathlib import Path

from .exp2_completion import validate_exp2_completion_artifact
from .exp2_protocol import expected_runs, validate_contract
from .experiment_manifest import MANIFEST_RELATIVE_PATH, load_manifest
from .provenance import (
    corrected_per_combo_artifact, corrected_prediction_artifact,
    prediction_truth_digest,
)


def validated_exp2_artifacts(repo, runs_dir=None):
    """Return `(run, metrics_path, payload)` only after the whole roster passes.

    There is deliberately no partial/unchecked mode: every paper consumer must
    compare the same arms, seeds, source targets and completed training recipes.
    Deterministic baselines still occupy their manifest-defined paired slots.
    This function does not write results or modify training artifacts.
    """
    repo = Path(repo)
    directory = repo / "outputs/full" if runs_dir is None else Path(runs_dir)
    manifest = load_manifest(repo / MANIFEST_RELATIVE_PATH)
    expected = {run.directory_name: run for run in expected_runs(manifest)}
    actual = {p.parent.name for p in directory.glob("*/metrics.json")}
    if actual != set(expected):
        raise RuntimeError(
            f"Inexact Exp 2 roster/manifest: missing {len(set(expected) - actual)}, "
            f"unexpected {len(actual - set(expected))} runs"
        )
    artifacts, truth = [], None
    identified = []
    for name, run in expected.items():
        path = directory / name / "metrics.json"
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict) or any(
                payload.get(key) != value or type(payload.get(key)) is not type(value)
                for key, value in (("model", run.model), ("variant", run.variant), ("seed", run.seed))
            ):
                raise ValueError("Exp 2 directory/identity mismatch: payload does not match the roster entry")
            identified.append((run, path, payload))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Refusing to publish {name}: {exc}") from exc
    for run, path, payload in identified:
        name = run.directory_name
        try:
            for check in (
                lambda: corrected_per_combo_artifact(payload),
                lambda: validate_contract(payload, repo),
            ):
                valid, reason = check()
                if not valid:
                    raise ValueError(reason)
            validate_exp2_completion_artifact(payload, path)
            valid, reason = corrected_prediction_artifact(payload, path, verify_scaled_metrics=True)
            if not valid:
                raise ValueError(reason)
            current = prediction_truth_digest(payload, path)
            if truth is not None and current != truth:
                raise ValueError("Exp 2 test targets differ across runs")
            truth = current
            artifacts.append((run, path, payload))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Refusing to publish {name}: {exc}") from exc
    return artifacts
