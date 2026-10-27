"""Every run's metrics.json must say whether the network was two-sided over
the input window, because the ND arms are and most 1-D arms are not. This pins
the per-model declaration and the payload shape scripts/inference.py writes."""
import numpy as np
import pytest
import torch

from src.models.gpt import make_gpt
from src.models.gru import make_gru
from src.models.lstm import make_lstm
from src.models.s4 import make_s4
from src.models.transformer import make_transformer

COMMON = dict(num_numeric_features=5, num_states=2, num_commodities=2, num_flows=2)


def _coords():
    return np.array([[s, c, f] for s in range(2) for c in range(2) for f in range(2)])


@pytest.mark.parametrize(("factory", "kwargs", "expected"), [
    (make_lstm, {"hidden_size": 8, "num_layers": 1}, False),
    (make_gru, {"hidden_size": 8, "num_layers": 1}, False),
    (make_gpt, {"d_model": 8, "nhead": 2, "mlp_ratio": 2.0, "num_layers": 1}, False),
    (make_s4, {"d_model": 16, "n_layers": 1, "d_state": 8}, False),
    # The vanilla Transformer encoder is unmasked: every step sees the whole
    # window, so it is the one two-sided 1-D arm in the benchmark.
    (make_transformer, {"hidden_size": 8, "num_heads": 2, "d_ff": 16, "num_layers": 1}, True),
])
def test_flat_models_declare_their_causality(factory, kwargs, expected):
    model = factory(variant="embeddings", **COMMON, **kwargs)
    prov = model.architecture_provenance()
    assert prov["bidirectional"] is expected
    assert prov["variant"] == "embeddings"
    assert prov["num_params"] == model.get_num_params() > 0
    assert isinstance(prov["architecture_versions"], dict)


def test_flat_lstm_bidirectional_config_is_reported():
    model = make_lstm(variant="embeddings", **COMMON, hidden_size=8, num_layers=1,
                      bidirectional=True)
    assert model.architecture_provenance()["bidirectional"] is True


def test_combo_recurrent_arms_are_causal():
    model = make_lstm(
        variant="cross_attention", **COMMON, num_combos=8, features_per_group=5,
        combo_coords=_coords(), lattice_dims=(2, 2, 2), hidden_size=8, num_layers=1,
    )
    assert model.architecture_provenance()["bidirectional"] is False


def test_transformer_categorical_attention_reports_its_encoder_revision():
    """The rerun's transformer_cross_attention is a different model from the
    recorded one; the version string is how a reader tells them apart."""
    model = make_transformer(variant="cross_attention", **COMMON, hidden_size=8,
                             num_heads=2, d_ff=16, num_layers=1)
    versions = model.architecture_provenance()["architecture_versions"]
    assert "categorical_attention_v2" in versions.values()


def test_provenance_is_json_serialisable():
    import json
    model = make_gru(variant="embeddings", **COMMON, hidden_size=8, num_layers=1)
    json.dumps(model.architecture_provenance())
