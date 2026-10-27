"""Restore and validate the data cohort associated with a checkpoint."""

import json
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

from src.utils.provenance import BENCHMARK_PROTOCOL_VERSION


def apply_checkpoint_cohort_config(
    config: MutableMapping[str, Any],
    checkpoint_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Apply the checkpoint artifact's cohort mode to an inference config.

    Artifacts written before ``cohort_from_train_only`` was introduced used
    the legacy full-timeline filter, so a missing field is intentionally
    interpreted as ``False``.
    """
    artifact_path = (
        Path(checkpoint_path).expanduser().resolve().parent
        / "artifacts"
        / "kept_combos.json"
    )
    if not artifact_path.exists():
        return None

    try:
        metadata = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read checkpoint cohort artifact {artifact_path}: {exc}"
        ) from exc

    if metadata.get("benchmark_protocol_version") != BENCHMARK_PROTOCOL_VERSION:
        raise ValueError(
            f"Checkpoint at {artifact_path} predates benchmark protocol "
            f"{BENCHMARK_PROTOCOL_VERSION}; retraining is required"
        )

    train_only = metadata.get("cohort_from_train_only", False)
    if not isinstance(train_only, bool):
        raise ValueError(
            f"Invalid cohort_from_train_only in {artifact_path}: "
            f"expected bool, got {train_only!r}"
        )

    data_config = config.setdefault("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("Inference config's data section must be a mapping")
    data_config["cohort_from_train_only"] = train_only
    metadata["_artifact_path"] = str(artifact_path)
    return metadata


def validate_checkpoint_cohort(
    kept_combos,
    metadata: Optional[Dict[str, Any]],
    *,
    source_fingerprint: Optional[str] = None,
) -> None:
    """Fail rather than evaluate against a different cohort or source data."""
    if metadata is None:
        if source_fingerprint is not None:
            raise ValueError(
                "Checkpoint is missing cohort/source provenance; retraining "
                f"under benchmark protocol {BENCHMARK_PROTOCOL_VERSION} is required"
            )
        return

    if source_fingerprint is not None:
        expected_fingerprint = metadata.get("source_fingerprint")
        if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
            raise ValueError(
                "Checkpoint cohort artifact is missing its source fingerprint; "
                f"retraining under benchmark protocol {BENCHMARK_PROTOCOL_VERSION} is required"
            )
        if expected_fingerprint != source_fingerprint:
            raise ValueError(
                "Inference source data does not match the checkpoint artifact "
                f"at {metadata.get('_artifact_path', 'kept_combos.json')}: expected "
                f"{expected_fingerprint}, got {source_fingerprint}"
            )

    if kept_combos is None:
        return

    actual = [list(combo) for combo in kept_combos]
    expected = metadata.get("combos")
    if expected is not None and actual != expected:
        raise ValueError(
            "Rebuilt inference cohort does not match the checkpoint artifact "
            f"at {metadata['_artifact_path']}: expected {len(expected)} combos, "
            f"got {len(actual)}. Use the same data and split configuration as "
            "training."
        )

    expected_count = metadata.get("num_combos")
    if expected_count is not None and len(actual) != int(expected_count):
        raise ValueError(
            "Rebuilt inference cohort size does not match the checkpoint "
            f"artifact at {metadata['_artifact_path']}: expected "
            f"{expected_count}, got {len(actual)}."
        )
