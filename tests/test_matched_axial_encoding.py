import itertools

import pytest
import torch

from src.models.combo_attention import MatchedAxialCategoricalEncoder
from src.models.gpt import make_gpt
from src.models.gru import make_gru
from src.models.lstm import make_lstm
from src.models.s4 import make_s4
from src.models.transformer import make_transformer


def full_coords():
    return torch.tensor(list(itertools.product(range(2), range(2), range(2))))


@pytest.mark.parametrize(
    ("dims", "expected_extra", "has_commodity", "has_flow"),
    [(2, 34, True, True), (3, 2, False, True), (4, 0, False, False)],
)
def test_matched_encoder_uses_only_non_structural_identities(
    dims, expected_extra, has_commodity, has_flow
):
    encoder = MatchedAxialCategoricalEncoder(
        features_per_group=5,
        combo_coords=full_coords(),
        dims=dims,
        num_commodities=2,
        num_flows=2,
    )
    x = torch.zeros(2, 4, 8, 5)
    out = encoder(x)

    assert out.shape == (2, 4, 8, 5 + expected_extra)
    assert (encoder.commodity_embedding is not None) is has_commodity
    assert (encoder.flow_embedding is not None) is has_flow
    assert not any("state" in name for name, _ in encoder.named_parameters())


def test_matched_2d_embeddings_follow_each_groups_static_coordinates():
    coords = full_coords()
    encoder = MatchedAxialCategoricalEncoder(
        features_per_group=1,
        combo_coords=coords,
        dims=2,
        num_commodities=2,
        num_flows=2,
        comm_embed_dim=1,
        flow_embed_dim=1,
    )
    with torch.no_grad():
        encoder.commodity_embedding.weight[:, 0] = torch.tensor([3.0, 7.0])
        encoder.flow_embedding.weight[:, 0] = torch.tensor([11.0, 13.0])

    out = encoder(torch.zeros(1, 2, 8, 1))[0, 0, :, 1:]
    expected = torch.stack(
        [torch.tensor([3.0, 7.0])[coords[:, 1]],
         torch.tensor([11.0, 13.0])[coords[:, 2]]],
        dim=-1,
    )
    assert torch.equal(out, expected)


MODEL_FACTORIES = [
    ("gru", make_gru, {"hidden_size": 8}),
    ("lstm", make_lstm, {"hidden_size": 8}),
    ("transformer", make_transformer, {"hidden_size": 8, "num_heads": 2, "d_ff": 16}),
    ("gpt", make_gpt, {"d_model": 8, "nhead": 2, "mlp_ratio": 2.0}),
    # S4 runs the genuine S4Block with the cauchy_naive fallback, so the SSM
    # arm of the ACA family is fully exercised on CPU. (Mamba-2 needs triton and
    # is covered separately with a stubbed mixer.)
    ("s4", make_s4, {"d_model": 16, "n_layers": 1, "d_state": 8}),
]


@pytest.mark.parametrize("family", ["cross_attention", "matched_axial"])
@pytest.mark.parametrize(("model_name", "factory", "model_kwargs"), MODEL_FACTORIES)
@pytest.mark.parametrize(("dims", "expected_input"), [(2, 39), (3, 7), (4, 5)])
def test_all_matched_axial_models_forward_backward(
    model_name, factory, model_kwargs, dims, expected_input, family
):
    coords = full_coords()
    model = factory(
        variant=f"{family}_{dims}d",
        num_numeric_features=5,
        num_states=2,
        num_commodities=2,
        num_flows=2,
        num_combos=8,
        features_per_group=5,
        combo_coords=coords,
        lattice_dims=(2, 2, 2),
        num_layers=1,
        dropout=0.0,
        **model_kwargs,
    )
    x = torch.randn(2, 4, 8, 5)
    out = model(x, group_mask=torch.ones(2, 8))

    assert out.shape == (2, 8, 2), model_name
    assert torch.isfinite(out).all()
    assert model.group_encoder.output_dim == expected_input

    out.square().mean().backward()
    embedding_params = [
        parameter for name, parameter in model.named_parameters()
        if "group_encoder" in name and "embedding" in name
    ]
    if dims == 4:
        assert embedding_params == []
    else:
        assert embedding_params
        assert all(parameter.grad is not None for parameter in embedding_params)
        assert all(torch.isfinite(parameter.grad).all() for parameter in embedding_params)


def _mamba_nd_with_stub_mixer(monkeypatch, **kwargs):
    """Build mamba_nd with the Mamba-2 mixer stubbed out.

    The real mixer needs triton/GPU, but what is under test here is the
    ENCODER, not the scan — stubbing keeps the information-protocol check
    runnable on CPU-only machines instead of skipping where it matters.
    """
    import importlib
    import sys

    import torch.nn as nn

    from src.models.base import ModelFactory
    from src.models import vendor_imports

    class _StubMixer(nn.Module):
        def __init__(self, d_model=128, **kw):
            super().__init__()
            self.d_model = d_model

        def forward(self, x):
            return x

    monkeypatch.setattr(vendor_imports, "load_mamba2", lambda: _StubMixer)
    module = kwargs.pop("_module", "src.models.mamba_nd")
    factory = kwargs.pop("_factory", "make_mamba_nd")
    monkeypatch.delitem(sys.modules, module, raising=False)
    # Re-importing re-registers the model; give it a throwaway registry copy so
    # a real (GPU) registration on the box survives this test.
    monkeypatch.setattr(ModelFactory, "_models", dict(ModelFactory._models))
    return getattr(importlib.import_module(module), factory)(**kwargs)


@pytest.mark.parametrize(("dims", "expected_input"), [(2, 39), (3, 7), (4, 5)])
def test_mamba_nd_matches_the_s4nd_information_protocol(
    monkeypatch, dims, expected_input
):
    """mamba_nd must see the same identities as s4nd at each rank: it is the
    third mechanism sharing the cross_attention_{2,3,4}d rank label, and it
    previously took state/comm/flow ids in forward and discarded them."""
    model = _mamba_nd_with_stub_mixer(
        monkeypatch,
        variant=f"cross_attention_{dims}d",
        num_numeric_features=5,
        num_states=2,
        num_commodities=2,
        num_flows=2,
        num_combos=8,
        features_per_group=5,
        combo_coords=full_coords(),
        lattice_dims=(2, 2, 2),
        d_model=32,
        n_layers=2,
        headdim=8,
    )
    assert isinstance(model.group_encoder, MatchedAxialCategoricalEncoder)
    assert model.group_encoder.output_dim == expected_input
    assert model.grid.in_proj.in_features == expected_input
    assert not any("state" in n for n, _ in model.group_encoder.named_parameters())


@pytest.mark.parametrize(
    ("schedule", "expected_reverse"),
    [(None, [False, False, True, True]), ("legacy", [False, True, False, True])],
)
def test_encoder_does_not_disturb_the_scan_schedule(
    monkeypatch, schedule, expected_reverse
):
    """The identity encoder sits in front of the scan and must not touch it.
    Default ("cyclic") gives Time and State both directions; "legacy" still
    reproduces the phase-locked pattern of the recorded runs."""
    kwargs = {} if schedule is None else {"scan_schedule": schedule}
    model = _mamba_nd_with_stub_mixer(
        monkeypatch,
        variant="cross_attention_2d",
        num_numeric_features=5,
        num_states=2,
        num_commodities=2,
        num_flows=2,
        num_combos=8,
        features_per_group=5,
        combo_coords=full_coords(),
        lattice_dims=(2, 2, 2),
        d_model=32,
        n_layers=4,
        headdim=8,
        **kwargs,
    )
    assert model.grid.layer_pos == [1, 2, 1, 2]
    assert model.grid.layer_reverse == expected_reverse


@pytest.mark.parametrize(("dims", "expected_input"), [(2, 39), (3, 7), (4, 5)])
def test_mamba_combo_axial_joins_the_aca_family(monkeypatch, dims, expected_input):
    """Mamba-2 + ACA: the axial family's missing member. Same axial module and
    matched encoder as gru/lstm/transformer/gpt/s4, only the temporal block
    differs — so it needs no scan-depth or direction budget at all."""
    model = _mamba_nd_with_stub_mixer(
        monkeypatch,
        _module="src.models.mamba",
        _factory="make_mamba",
        variant=f"cross_attention_{dims}d",
        num_numeric_features=5,
        num_states=2,
        num_commodities=2,
        num_flows=2,
        num_combos=8,
        features_per_group=5,
        combo_coords=full_coords(),
        lattice_dims=(2, 2, 2),
        d_model=16,
        n_layers=1,
        dropout=0.0,
    )
    assert isinstance(model.group_encoder, MatchedAxialCategoricalEncoder)
    assert model.axial.in_proj.in_features == expected_input
    out = model(torch.randn(2, 4, 8, 5), group_mask=torch.ones(2, 8))
    assert out.shape == (2, 8, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("dims", "fixed_layers", "coverage_layers"), [(2, 4, 4), (3, 4, 6), (4, 4, 8)]
)
def test_mamba_nd_depth_policies(monkeypatch, dims, fixed_layers, coverage_layers):
    """"fixed" keeps the iso-layer 4; "coverage" scales depth with the axis
    count so every rank gets the same per-axis budget (one forward + one
    backward). They are an ablation pair — both get run and reported."""
    def build(policy):
        return _mamba_nd_with_stub_mixer(
            monkeypatch,
            variant=f"cross_attention_{dims}d",
            depth_policy=policy,
            num_numeric_features=5,
            num_states=2,
            num_commodities=2,
            num_flows=2,
            num_combos=8,
            features_per_group=5,
            combo_coords=full_coords(),
            lattice_dims=(2, 2, 2),
            d_model=32,
            n_layers=4,
            headdim=8,
        )

    assert len(build("fixed").grid.layer_pos) == fixed_layers
    coverage = build("coverage")
    assert len(coverage.grid.layer_pos) == coverage_layers
    # Every axis gets exactly one forward and one backward pass.
    seen = {}
    for pos, rev in zip(coverage.grid.layer_pos, coverage.grid.layer_reverse):
        seen.setdefault(pos, []).append(rev)
    assert len(seen) == dims, "every axis must be scanned"
    assert all(sorted(v) == [False, True] for v in seen.values())


def test_mamba_nd_default_depth_is_capacity_matched(monkeypatch):
    model = _mamba_nd_with_stub_mixer(
        monkeypatch,
        variant="cross_attention_4d",
        num_numeric_features=5,
        num_states=2,
        num_commodities=2,
        num_flows=2,
        num_combos=8,
        features_per_group=5,
        combo_coords=full_coords(),
        lattice_dims=(2, 2, 2),
        d_model=32,
        n_layers=4,
        headdim=8,
    )
    assert model.n_layers == 4


def test_mamba_nd_compacts_sparse_cells_before_scan(monkeypatch):
    """An absent cell must not count as a recurrent transition."""
    model = _mamba_nd_with_stub_mixer(
        monkeypatch,
        variant="cross_attention_2d",
        depth_policy="fixed",
        num_numeric_features=1,
        num_states=3,
        num_commodities=1,
        num_flows=1,
        num_combos=2,
        features_per_group=1,
        combo_coords=torch.tensor([[0, 0, 0], [2, 0, 0]]),
        lattice_dims=(3, 1, 1),
        d_model=1,
        n_layers=2,
        headdim=1,
        dropout=0.0,
    )

    class DecayingScan(torch.nn.Module):
        def forward(self, x):
            state = torch.zeros_like(x[:, 0])
            out = []
            for i in range(x.shape[1]):
                state = 0.5 * state + x[:, i]
                out.append(state)
            return torch.stack(out, dim=1)

    # Exercise the State-axis scan directly with an artificial gap between two
    # real cells. LayerNorm is intentionally bypassed so the expected recurrence
    # is readable.
    dense = torch.tensor(
        [[[[[[1.0]]], [[[0.0]]], [[[2.0]]]]]], requires_grad=True
    )
    out = model.grid._scan_axis(
        dense, pos=2, mixer=DecayingScan(), norm=torch.nn.Identity(), reverse=False
    )
    # Compact sequence [1, 2] gives mixer outputs [1, 2.5], then residuals
    # [2, 4.5]. Scanning [1, 0, 2] would incorrectly produce 4.25 at the end.
    assert out[0, 0, 0, 0, 0, 0].item() == pytest.approx(2.0)
    assert out[0, 0, 1, 0, 0, 0].item() == 0.0
    assert out[0, 0, 2, 0, 0, 0].item() == pytest.approx(4.5)
    out.sum().backward()
    assert torch.isfinite(dense.grad).all()


def test_sparse_lattice_has_finite_forward_and_backward():
    coords = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 1]])
    model = make_gru(
        variant="matched_axial_2d",
        num_numeric_features=3,
        num_states=2,
        num_commodities=2,
        num_flows=2,
        num_combos=3,
        features_per_group=3,
        combo_coords=coords,
        lattice_dims=(2, 2, 2),
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    out = model(torch.randn(2, 4, 3, 3), group_mask=torch.ones(2, 3))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_cross_attention_variants_use_the_matched_protocol():
    """The ACA variants ARE the matched protocol: cross_attention_{2,3,4}d and
    matched_axial_{2,3,4}d must build identical encoders. Feeding the backbone
    raw features only (the old behaviour) starved cross_attention_2d of the
    commodity/flow identities s4nd_2d re-injects via its cfid encoder."""
    shared = dict(
        num_numeric_features=5,
        num_states=2,
        num_commodities=2,
        num_flows=2,
        num_combos=8,
        features_per_group=5,
        combo_coords=full_coords(),
        lattice_dims=(2, 2, 2),
        hidden_size=8,
        num_layers=1,
    )
    aca = make_gru(variant="cross_attention_2d", **shared)
    alias = make_gru(variant="matched_axial_2d", **shared)

    assert isinstance(aca.group_encoder, MatchedAxialCategoricalEncoder)
    assert aca.group_encoder.output_dim == alias.group_encoder.output_dim == 39
    assert aca.axial.in_proj.in_features == 39
    # The original name survives for checkpoints / results dirs.
    assert aca.variant == "cross_attention_2d"
