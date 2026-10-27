from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _yaml(path):
    return yaml.safe_load((ROOT / path).read_text())


def test_all_benchmark_families_use_the_shared_learning_rate():
    base = _yaml("config/base.yaml")
    shared = float(base["training"]["lr"])
    assert shared == 1e-4
    assert base["data"]["tabular_align_targets"] is True
    assert "training" not in _yaml("config/models/transformer.yaml")

    for name in (
        "s4_embeddings", "s4_onehot", "s4nd_2d", "s4nd_2d_state",
        "s4nd_3d", "s4nd_4d",
    ):
        cfg = _yaml(f"external/s4/configs/experiment/{name}.yaml")
        assert float(cfg["optimizer"]["lr"]) == shared, name


def test_mamba_nd_primary_protocol_is_fixed_depth():
    cfg = _yaml("config/models/mamba_nd.yaml")
    assert cfg["model"]["n_layers"] == 4
    assert cfg["model"]["depth_policy"] == "fixed"


def test_s4_backbones_share_one_state_size():
    """The s4nd-vs-s4+ACA pair exists to hold the sequence model fixed and vary
    only how the grid is injected; a 64-vs-128 state size broke that pair
    (docs/FAIRNESS_AUDIT.md). Settled on 64 for the whole family, 2026-09-13."""
    local = _yaml("config/models/s4.yaml")["model"]["d_state"]
    assert local == 64
    for name in (
        "s4_embeddings", "s4_onehot", "s4nd_2d", "s4nd_2d_state",
        "s4nd_3d", "s4nd_4d",
    ):
        cfg = _yaml(f"external/s4/configs/experiment/{name}.yaml")
        assert cfg["model"]["layer"]["d_state"] == local, name


def test_exp1_trains_on_every_window_in_equal_steps():
    """132 aggregate windows: batch 64 + drop_last silently trained on a
    different 128-window subset every epoch. Batch 33 gives four equal steps
    over all 132 targets, and the
    aggregate path never drops a partial batch."""
    exp1 = _yaml("config/exp1.yaml")
    assert exp1["training"]["batch_size"] == 33
    assert 132 % exp1["training"]["batch_size"] == 0
    assert exp1["data"]["train_drop_last"] is False


def test_causality_confound_is_declared_not_hidden():
    """ND arms are two-sided over the input window, every 1-D comparator is
    causal. The configs may keep that design, but it has to be visible."""
    assert _yaml("config/models/mamba_nd.yaml")["model"]["bidirectional"] is True
    for name in ("s4nd_2d", "s4nd_2d_state", "s4nd_3d", "s4nd_4d"):
        assert _yaml(f"external/s4/configs/experiment/{name}.yaml")["model"]["layer"]["bidirectional"] is True, name
    for name in ("s4_embeddings", "s4_onehot"):
        assert _yaml(f"external/s4/configs/experiment/{name}.yaml")["model"]["layer"]["bidirectional"] is False, name
    for name in ("lstm", "gru"):
        assert _yaml(f"config/models/{name}.yaml")["model"]["bidirectional"] is False, name
    # the two-sided 1-D ablation that closes the comparison must exist
    bidir = _yaml("external/s4/configs/experiment/s4_embeddings_bidir.yaml")
    assert bidir["model"]["layer"]["bidirectional"] is True
    assert bidir["model"]["layer"]["d_state"] == 64
