"""Linear-head Transformer/GPT rerun: heads, roster, contracts and isolation."""
import itertools
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.models import create_model
from src.utils.config import load_config
from src.utils.experiment_manifest import exp1_expected_runs, exp1_protocol, exp1_source_domain
from src.utils.exp2_protocol import expected_contract, expected_runs, validate_contract
from src.utils.linhead_protocol import LINHEAD_MODELS, exp1_runs, exp2_runs, manifest
from src.utils.run_contract import expected_exp1_run_contract

ROOT = Path(__file__).resolve().parents[1]
COORDS = torch.tensor(list(itertools.product(range(2), range(2), range(2))))


def _build(name, variant, **kw):
    cfg = load_config(str(ROOT / f"config/models/{name}.yaml"))["model"]
    cfg = {k: v for k, v in cfg.items() if k != "name"}
    cfg.update(num_layers=1, **kw)
    if variant.startswith("cross_attention_"):
        cfg.update(num_combos=8, features_per_group=5, combo_coords=COORDS, lattice_dims=(2, 2, 2))
    return create_model(name, variant, 5, 2, 2, **cfg)


def _head(model):
    return model.output_layer if hasattr(model, "output_layer") else model.head


@pytest.mark.parametrize("base", sorted(LINHEAD_MODELS))
@pytest.mark.parametrize("variant", ["embeddings", "cross_attention_4d"])
def test_linhead_is_one_linear_and_published_head_is_unchanged(base, variant):
    linear = _head(_build(LINHEAD_MODELS[base], variant))
    assert isinstance(linear, nn.Linear) and (linear.in_features, linear.out_features) == (128, 2)
    mlp = _head(_build(base, variant))
    hidden = {"transformer": 64, "gpt": 128}[base]
    assert [tuple(m.weight.shape) for m in mlp if isinstance(m, nn.Linear)] == [(hidden, 128), (2, hidden)]


@pytest.mark.parametrize("base", sorted(LINHEAD_MODELS))
def test_only_the_head_differs(base):
    torch.manual_seed(0)
    a = _build(base, "embeddings").state_dict()
    b = _build(LINHEAD_MODELS[base], "embeddings").state_dict()
    head = "output_layer." if base == "transformer" else "head."
    strip = lambda sd: {k: tuple(v.shape) for k, v in sd.items() if not k.startswith(head)}
    assert strip(a) == strip(b)


def test_transformer_cross_and_film_variants_take_the_linear_head():
    for variant in ("cross_attention", "film_attention"):
        cfg = load_config(str(ROOT / "config/models/transformer_linhead.yaml"))["model"]
        cfg = {k: v for k, v in cfg.items() if k != "name"}
        m = create_model("transformer_linhead", variant, 5, 2, 2, **{**cfg, "num_layers": 1})
        assert isinstance(m.output_layer, nn.Linear)


def test_roster_twins_every_published_transformer_gpt_run_exactly():
    m = manifest()
    published1 = {(r.model, r.variant, r.seed) for r in exp1_expected_runs(m) if r.model in LINHEAD_MODELS}
    assert {(LINHEAD_MODELS[a], v, s) for a, v, s in published1} == {(r.model, r.variant, r.seed) for r in exp1_runs(m)}
    published2 = {(r.model, r.variant, r.seed) for r in expected_runs(m)
                  if r.kind == "native" and r.model in LINHEAD_MODELS}
    assert {(LINHEAD_MODELS[a], v, s) for a, v, s in published2} == {(r.model, r.variant, r.seed) for r in exp2_runs(m)}
    assert (len(exp1_runs(m)), len(exp2_runs(m))) == (100, 36)


def test_contract_differs_from_published_only_in_name_and_head():
    m = manifest()
    pub = next(r for r in expected_runs(m) if (r.model, r.variant) == ("gpt", "cross_attention_4d"))
    lin = next(r for r in exp2_runs(m) if (r.model, r.variant, r.seed) == ("gpt_linhead", pub.variant, pub.seed))
    a, b = expected_contract(ROOT, pub, m), expected_contract(ROOT, lin, m)

    def diff(x, y, path=()):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in set(x) | set(y):
                yield from diff(x.get(k), y.get(k), path + (k,))
        elif x != y:
            yield path
    paths = {p[-1] for p in diff(a, b)}
    assert paths <= {"model", "name", "prediction_head"}, paths
    assert paths & {"prediction_head"}


def test_exp1_contract_reconstructs_for_linhead_runs():
    m = manifest()
    run = exp1_runs(m)[0]
    contract = expected_exp1_run_contract(ROOT, run, exp1_protocol(m), exp1_source_domain(m))
    assert "prediction_head" in str(contract)


def test_published_gate_rejects_linhead_and_linhead_gate_rejects_published():
    m = manifest()
    pub = next(r for r in expected_runs(m) if (r.model, r.variant) == ("transformer", "embeddings"))
    lin = next(r for r in exp2_runs(m) if (r.model, r.variant, r.seed) == ("transformer_linhead", "embeddings", pub.seed))
    lin_payload = {"model": lin.model, "variant": lin.variant, "seed": lin.seed,
                   "run_contract": expected_contract(ROOT, lin, m)}
    pub_payload = {"model": pub.model, "variant": pub.variant, "seed": pub.seed,
                   "run_contract": expected_contract(ROOT, pub, m)}
    assert validate_contract(lin_payload, ROOT, exp2_runs) == (True, "current")
    assert validate_contract(lin_payload, ROOT)[0] is False
    assert validate_contract(pub_payload, ROOT, exp2_runs)[0] is False
    assert validate_contract(pub_payload, ROOT) == (True, "current")
    # A linear-head payload claiming the MLP-head contract is stale.
    assert validate_contract({**lin_payload, "run_contract": pub_payload["run_contract"]}, ROOT, exp2_runs)[0] is False
