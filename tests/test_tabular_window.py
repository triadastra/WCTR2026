"""Matched-history trees must receive exactly the neural samples."""
import numpy as np
import pytest
from copy import deepcopy
from pathlib import Path

from tests.test_tabular_target_alignment import _pipe, _loaders, _tabular
from src.utils.config import apply_legacy_reproduction, load_config
from src.utils.run_contract import validate_run_contract


ROOT = Path(__file__).resolve().parents[1]
TREE_MODELS = ("xgboost", "lightgbm", "random_forest")


@pytest.mark.parametrize("experiment", [1, 2])
def test_canonical_tree_contracts_require_full_history(experiment):
    from src.utils.experiment_manifest import (
        load_manifest, exp1_expected_runs, exp1_protocol, exp1_source_domain,
    )
    from src.utils.exp2_protocol import expected_runs, expected_contract
    from src.utils.run_contract import expected_exp1_run_contract

    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    runs = exp1_expected_runs(manifest) if experiment == 1 else expected_runs(manifest)
    checked = set()
    for run in runs:
        if run.model not in TREE_MODELS:
            continue
        cfg = load_config(str(ROOT / "config/base.yaml"))
        for path in (f"config/models/{run.model}.yaml", f"config/variants/{run.variant}.yaml"):
            cfg = load_config(str(ROOT / path), cfg)
        if experiment == 1:
            cfg = load_config(str(ROOT / "config/exp1.yaml"), cfg)
            cfg["data"]["aggregate"] = True
            contract = expected_exp1_run_contract(
                ROOT, run, exp1_protocol(manifest), exp1_source_domain(manifest),
            )
        else:
            contract = expected_contract(ROOT, run, manifest)
        assert cfg["data"]["tabular_feature_mode"] == "window"
        assert contract["data_config"]["tabular_feature_mode"] == "window"
        identity = dict(contract["identity"])
        identity.update(source_fingerprint=contract["source_fingerprint"],
                        source_domain=contract["source_domain"])
        validate_run_contract(contract, cfg, **identity)
        for old_mode in ("last_row", None):
            stale = deepcopy(contract)
            if old_mode is None:
                stale["data_config"].pop("tabular_feature_mode")
            else:
                stale["data_config"]["tabular_feature_mode"] = old_mode
            with pytest.raises(ValueError, match="data_config"):
                validate_run_contract(stale, cfg, **identity)
        legacy = apply_legacy_reproduction(cfg, aggregate=experiment == 1)
        assert legacy["data"]["tabular_feature_mode"] == "last_row"
        checked.add((run.model, run.variant))
    expected_models = {"xgboost", "lightgbm"} if experiment == 1 else set(TREE_MODELS)
    assert checked == {(model, variant) for model in expected_models
                       for variant in ("embeddings", "onehot")}


@pytest.mark.parametrize("model_name", TREE_MODELS)
@pytest.mark.parametrize("variant", ["embeddings", "onehot"])
def test_configured_window_tree_fit_predict_and_reload(model_name, variant, tmp_path):
    from importlib import import_module

    classes = {"xgboost": "XGBoostModel", "lightgbm": "LightGBMModel",
               "random_forest": "RandomForestModel"}
    cls = getattr(import_module(f"src.models.{model_name}"), classes[model_name])
    cfg = load_config(str(ROOT / f"config/models/{model_name}.yaml"))
    pipe = _pipe(True)
    pipe.config.tabular_feature_mode = cfg["data"]["tabular_feature_mode"]
    train, val, test = _loaders(pipe)
    kwargs = dict(cfg["model"])
    kwargs.pop("name")
    kwargs.update(n_estimators=2, n_jobs=1, max_depth=2, random_state=26)
    dimensions = dict(variant=variant, num_numeric_features=len(pipe.feat_cols),
                      num_states=len(pipe.state2id), num_commodities=len(pipe.comm2id),
                      num_flows=len(pipe.flow2id))
    model = cls(**dimensions, **kwargs)
    model.fit(train, val, verbose=False)
    prediction, target = model.predict_loader(test)
    assert prediction.shape == target.shape == (len(test.dataset), 2)
    assert np.isfinite(prediction).all()
    width = pipe.config.input_len * len(pipe.feat_cols)
    assert len(model.feature_names) > width
    assert model.feature_names[0].startswith(f"history_{pipe.config.input_len}__")
    expected_target = np.array([[s["target_value"], s["target_weight"]]
                                for s in test.dataset.samples])
    np.testing.assert_array_equal(target, expected_target)
    path = str(tmp_path / "tree.pkl")
    model.save(path)
    restored = cls(**dimensions, **kwargs)
    restored.load(path)
    reloaded, reloaded_target = restored.predict_loader(test)
    np.testing.assert_array_equal(reloaded, prediction)
    np.testing.assert_array_equal(reloaded_target, target)


@pytest.mark.parametrize("split", [0, 1, 2])
def test_window_features_and_targets_equal_neural_samples(split):
    pipe = _pipe(True)
    pipe.config.tabular_feature_mode = "window"
    loader = _loaders(pipe)[split]
    x, y, names = _tabular(pipe, loader)
    samples = loader.dataset.samples
    numeric_width = pipe.config.input_len * len(pipe.feat_cols)
    assert len(x) == len(samples)
    assert len(names) == x.shape[1] == numeric_width + 3
    for i, sample in enumerate(samples):
        np.testing.assert_array_equal(
            x[i, :numeric_width], sample["x_numeric"].numpy().reshape(-1)
        )
        np.testing.assert_array_equal(
            y[i], [sample["target_value"], sample["target_weight"]]
        )
        np.testing.assert_array_equal(
            x[i, numeric_width:],
            [sample["state_id"], sample["comm_id"], sample["flow_id"]],
        )
    assert names[-3:] == ["state_id", "comm_id", "flow_id"]


def test_window_extraction_keeps_partial_batch_and_ignores_shuffle():
    pipe = _pipe(True)
    pipe.config.tabular_feature_mode = "window"
    ordered = pipe.get_dataloaders(7, num_workers=0, shuffle_train=False)[0]
    shuffled = pipe.get_dataloaders(11, num_workers=0, shuffle_train=True)[0]
    assert len(ordered.dataset) % 7 != 0
    x, y, names = _tabular(pipe, ordered)
    other_x, other_y, other_names = _tabular(pipe, shuffled)
    assert len(x) == len(ordered.dataset)
    np.testing.assert_array_equal(x, other_x)
    np.testing.assert_array_equal(y, other_y)
    assert names == other_names


def test_invalid_feature_mode_rejected():
    pipe = _pipe(True)
    pipe.config.tabular_feature_mode = "typo"
    with pytest.raises(ValueError, match="tabular_feature_mode"):
        _loaders(pipe)


def test_window_onehot_preserves_numeric_history_and_category_indicators():
    from src.models.tabular_common import extract_tabular_xy
    pipe = _pipe(True)
    pipe.config.tabular_feature_mode = "window"
    loader = _loaders(pipe)[0]
    x, y, names = extract_tabular_xy(
        loader, variant="onehot", num_numeric_features=len(pipe.feat_cols),
        num_states=len(pipe.state2id), num_commodities=len(pipe.comm2id),
        num_flows=len(pipe.flow2id),
    )
    width = pipe.config.input_len * len(pipe.feat_cols)
    np.testing.assert_array_equal(x[:, width:].sum(axis=1), np.full(len(x), 3))
    assert set(np.unique(x[:, width:])) <= {0, 1}
    np.testing.assert_array_equal(x[0, :width], loader.dataset.samples[0]["x_numeric"].numpy().reshape(-1))
    assert len(names) == x.shape[1]
