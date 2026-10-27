from types import SimpleNamespace

import pytest

from src.utils.s4_checkpoint_provenance import (
    checkpoint_cohort_metadata,
    validate_checkpoint_cohort,
)


def _dataset(combos, train_only=True):
    pipe = SimpleNamespace(
        kept_combos=combos,
        source_fingerprint="a" * 64,
        source_domain={"states": ["A", "B"], "grid_start": "2009-01-01"},
        config=SimpleNamespace(
            cohort_from_train_only=train_only,
            input_len=36,
            lag_count=12,
            train_target_start=None,
        ),
    )
    return SimpleNamespace(pipe=pipe)


def test_s4_checkpoint_records_and_validates_exact_training_cohort():
    dataset = _dataset([("A", "C", "Import"), ("B", "D", "Export")])
    metadata = checkpoint_cohort_metadata(dataset)

    assert validate_checkpoint_cohort({"cohort": metadata}, dataset) == metadata


def test_s4_checkpoint_rejects_missing_or_different_training_cohort():
    dataset = _dataset([("A", "C", "Import")])
    with pytest.raises(RuntimeError, match="no training-time cohort provenance"):
        validate_checkpoint_cohort({}, dataset)

    metadata = checkpoint_cohort_metadata(dataset)
    changed = _dataset([("B", "C", "Import")])
    with pytest.raises(RuntimeError, match="does not match"):
        validate_checkpoint_cohort({"cohort": metadata}, changed)

    changed_source = _dataset([("A", "C", "Import")])
    changed_source.pipe.source_fingerprint = "b" * 64
    with pytest.raises(RuntimeError, match="source_fingerprint"):
        validate_checkpoint_cohort({"cohort": metadata}, changed_source)


def test_s4_checkpoint_rejects_legacy_full_timeline_training():
    dataset = _dataset([("A", "C", "Import")], train_only=False)
    metadata = checkpoint_cohort_metadata(dataset)
    with pytest.raises(RuntimeError, match="not trained on a train-only cohort"):
        validate_checkpoint_cohort({"cohort": metadata}, dataset)
