"""Pinned-source regression tests: no training and no source-file writes."""
import json
import hashlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from src.data.dataloader import DataConfig, TradeDataPipeline, KEYS
from src.utils.config import (
    apply_legacy_reproduction, get_data_config, load_config, require_legacy_output,
)
from src.utils.experiment_manifest import (
    exp1_source_domain, load_manifest, validate_exp1_config,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA256 = "55b96c34bb69f131a3a1c419c1e48f1112b2cd93c48e312c8f1aa21ad64327f4"


def config():
    cfg = apply_legacy_reproduction(load_config(str(ROOT / "config/base.yaml")))
    data = get_data_config(cfg)
    for flow in ("imports", "exports"):
        data[f"{flow}_dir"] = str(ROOT / "data" / flow)
    return data


@pytest.fixture(scope="module")
def loaded():
    pipe = TradeDataPipeline(DataConfig(**config()))
    with pytest.warns(RuntimeWarning, match="Legacy reproduction"):
        pipe.load_data()
    return pipe


def test_exact_historical_raw_grid_and_cohort(loaded):
    source = (ROOT / "tests/fixtures/legacy_dataloader_c5ce1b9.py.txt").read_bytes()
    assert hashlib.sha256(source).hexdigest() == REFERENCE_SHA256
    module = types.ModuleType("legacy_reference_dataloader")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "historical_dataloader.py", "exec"), module.__dict__)
        old = module.TradeDataPipeline(module.DataConfig(
            imports_dir=str(ROOT / "data/imports"),
            exports_dir=str(ROOT / "data/exports"), cohort_from_train_only=False,
        ))
        old.load_data()
        sort = KEYS + ["Country", "Time", "Value", "Weight"]
        pd.testing.assert_frame_equal(
            loaded.df_full.sort_values(sort).reset_index(drop=True),
            old.df_full.sort_values(sort).reset_index(drop=True), check_exact=True,
        )
        actual = loaded.build_grid(loaded.df_full)
        expected = old.build_grid(old.df_full)
        sort = KEYS + ["Time"]
        pd.testing.assert_frame_equal(
            actual.sort_values(sort).reset_index(drop=True),
            expected.sort_values(sort).reset_index(drop=True), check_exact=True,
        )
        assert not actual.duplicated(sort).any()
        a = loaded.filter_sparse_combos(actual)
        b = old.filter_sparse_combos(expected)
        pd.testing.assert_frame_equal(
            a.sort_values(sort).reset_index(drop=True),
            b.sort_values(sort).reset_index(drop=True), check_exact=True,
        )
        assert len(loaded.kept_combos) == 903
    finally:
        sys.modules.pop(module.__name__, None)


def test_audit_is_persisted(loaded, tmp_path):
    assert loaded.duplicate_audit == {
        "policy": "legacy_sum", "groups": 2250, "rows": 4500,
        "excess_rows": 2250, "conflicting_target_groups": 2198,
    }
    loaded.save_artifacts(str(tmp_path))
    saved = json.loads((tmp_path / "artifacts/source_data.json").read_text())
    assert saved["source_domain"]["duplicate_audit"] == loaded.duplicate_audit


@pytest.mark.parametrize("change,match", [
    ({"duplicate_policy": "error", "allow_legacy_full_timeline": False,
      "cohort_from_train_only": True}, "ambiguous duplicate"),
    ({"expected_source_fingerprint": "0" * 64}, "fingerprint mismatch"),
])
def test_strict_and_source_guards(change, match):
    with pytest.raises(ValueError, match=match):
        TradeDataPipeline(DataConfig(**{**config(), **change})).load_data()


def test_unpinned_legacy_rejected():
    with pytest.raises(ValueError, match="pinned source fingerprint"):
        DataConfig(target_measure="legacy_mixed", duplicate_policy="legacy_sum")


def test_pinned_source_does_not_implicitly_enable_holdout_cohort_selection():
    cfg = config()
    cfg["allow_legacy_full_timeline"] = False
    pipe = TradeDataPipeline(DataConfig(**cfg))
    with pytest.raises(ValueError, match="cohort_from_train_only must be True"):
        pipe.filter_sparse_combos(pd.DataFrame())
    with pytest.raises(ValueError, match="cohort_from_train_only must be True"):
        pipe.create_splits()


@pytest.mark.parametrize("change", [
    {"cohort_from_train_only": True},
    {"duplicate_policy": "error"},
])
def test_full_timeline_opt_in_requires_a_consistent_legacy_profile(change):
    with pytest.raises(ValueError, match="explicit pinned legacy reproduction profile"):
        DataConfig(**{**config(), **change})


def test_aggregate_dataloader_targets_are_unique_and_finite(loaded):
    import torch
    pipe = TradeDataPipeline(DataConfig(**{**config(), "aggregate": True}))
    pipe.df_full = loaded.df_full
    pipe.create_splits()
    train, val, test = pipe.get_dataloaders(batch_size=66, num_workers=0)
    assert len(train.dataset) == 132
    assert len(val.dataset) == 12
    assert len(test.dataset) == 5
    for loader in (train, val, test):
        for batch in loader:
            assert torch.isfinite(batch["target_value"]).all()
            assert torch.isfinite(batch["target_weight"]).all()


def test_profile_preserves_aggregate_batch_and_isolates_output():
    base = load_config(str(ROOT / "config/base.yaml"))
    base["training"]["batch_size"] = 66
    assert apply_legacy_reproduction(base, aggregate=True)["training"]["batch_size"] == 66
    assert apply_legacy_reproduction(base)["training"]["batch_size"] == 903
    require_legacy_output(str(ROOT / "outputs/legacy_reproduction/test_run"))
    with pytest.raises(ValueError, match="Legacy output"):
        require_legacy_output(str(ROOT / "outputs/full/existing_run"))


def test_canonical_monthly_release_preserves_original_targets(loaded):
    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    cfg = load_config(str(ROOT / "config/base.yaml"))
    cfg = load_config(str(ROOT / "config/exp1.yaml"), cfg)
    validate_exp1_config(manifest, cfg)
    monthly = TradeDataPipeline(DataConfig(**get_data_config(cfg)))
    monthly.load_data()
    assert exp1_source_domain(manifest) == monthly.source_domain
    keys = ["State", "Commodity", "Import/Export", "Time"]
    # Materializing the sum changes provenance, not any monthly target.
    original = loaded.build_grid(loaded.df_full).set_index(keys)[["Value", "Weight"]].sort_index()
    actual = monthly.build_grid(monthly.df_full).set_index(keys)[["Value", "Weight"]].sort_index()
    pd.testing.assert_frame_equal(actual, original, check_exact=True)
    assert monthly.source_fingerprint != loaded.source_fingerprint
    assert cfg["data"]["target_measure"] == "legacy_mixed"
    assert cfg["data"]["duplicate_policy"] == "error"


@pytest.mark.parametrize("field", ["expected_states", "expected_hs2_codes", "duplicate_policy"])
def test_canonical_exp1_cannot_remove_domain_or_duplicate_guards(field):
    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    cfg = load_config(str(ROOT / "config/base.yaml"))
    cfg = load_config(str(ROOT / "config/exp1.yaml"), cfg)
    cfg["data"].pop(field)
    with pytest.raises(ValueError):
        validate_exp1_config(manifest, cfg)
