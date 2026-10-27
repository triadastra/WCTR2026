"""All canonical S4 arms use the actual authors' modular implementation."""
import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.models.s4 import make_s4
from src.models.s4_modular import (
    IMPLEMENTATION, backbone_config, modular_s4_imports,
)
from src.utils.exp2_protocol import Exp2Run, expected_contract, resolved_s4_config
from src.utils.exp3_protocol import expected_contract as exp3_contract
from src.utils.exp3_protocol import coordinates, resolved_config
from src.utils.config import load_config
from src.utils.run_contract import build_run_contract, validate_run_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_model(variant="embeddings", **extra):
    coords = np.array([[s, c, f] for s in range(2) for c in range(2) for f in range(2)])
    return make_s4(variant, num_numeric_features=4, num_states=2, num_commodities=2,
                   num_flows=2, d_model=16, n_layers=2, d_state=8, dropout=0.1,
                   num_combos=8, features_per_group=4, combo_coords=coords,
                   lattice_dims=(2, 2, 2), **extra)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_recipe_is_identical_to_external_exp2_backbone(bidirectional):
    model = "s4_bidir" if bidirectional else "s4"
    experiment = "s4_embeddings_bidir" if bidirectional else "s4_embeddings"
    cfg = resolved_s4_config(ROOT, Exp2Run(model, "embeddings", 26, "s4", experiment))
    expected = cfg["model"]
    expected.pop("encoder"); expected.pop("decoder")
    assert backbone_config(bidirectional=bidirectional) == expected


@pytest.mark.parametrize("variant", [
    "embeddings", "onehot", "cross_attention_2d", "cross_attention_3d",
    "cross_attention_4d", "matched_axial_2d", "matched_axial_3d", "matched_axial_4d",
])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_every_s4_variant_uses_modular_forward_backward_and_checkpoint(variant, bidirectional, tmp_path):
    model = small_model(variant, bidirectional=bidirectional)
    assert model.architecture_provenance()["bidirectional"] is bidirectional
    assert type(model.backbone).__name__ == "SequenceModel"
    assert model.backbone.architecture_version == IMPLEMENTATION
    assert isinstance(model.backbone.norm.norm, torch.nn.LayerNorm)
    for layer in model.backbone.layers:
        assert layer.layer.layer.bidirectional is bidirectional
        kernel = layer.layer.layer.kernel
        assert type(kernel).__name__ == "SSMKernelDPLR"
        assert kernel.n_ssm == 1
        assert kernel.dt_transform == "softplus"
        # The canonical recipe ties residual dropout, not FFT kernel dropout.
        assert layer.drop.tie is True
        for module in (model.backbone, layer.layer, layer.layer.layer, kernel):
            source = Path(module.forward.__func__.__code__.co_filename).resolve()
            assert source.is_relative_to(ROOT / "external/s4/src/models/sequence")
    is_combo = model.requires_combo_loader
    x = torch.randn((2, 5, 8, 4) if is_combo else (2, 5, 4))
    ids = torch.tensor([0, 1])
    target = torch.randn((2, 8, 2) if is_combo else (2, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    result = model(x, ids, ids, ids)
    assert result.shape == target.shape
    (result-target).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None for p in model.backbone.parameters())
    optimizer.step()
    model.eval()
    with torch.no_grad(): expected = model(x, ids, ids, ids)
    path = tmp_path / "state.pth"
    torch.save(model.state_dict(), path)
    restored = small_model(variant, bidirectional=bidirectional).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad(): actual = restored(x, ids, ids, ids)
    torch.testing.assert_close(actual, expected)


def test_constructor_restores_native_modules_on_success_and_failure():
    import src.data.dataloader  # noqa: F401 -- populate the namespace being tested
    before = {k: v for k, v in sys.modules.copy().items() if k == "src" or k.startswith("src.")}
    first = small_model()
    assert all(sys.modules[k] is v for k, v in before.items())
    with pytest.raises(RuntimeError, match="probe"):
        with modular_s4_imports():
            raise RuntimeError("probe")
    assert all(sys.modules[k] is v for k, v in before.items())
    second = small_model()
    assert first.get_num_params() == second.get_num_params()
    assert all(sys.modules[k] is v for k, v in before.items())
    # The registry still belongs to the benchmark, not to external S4.
    from src.models.base import ModelFactory
    assert "lstm" in ModelFactory.list_models()
    assert "s4" in ModelFactory.list_models()


def test_unknown_implementation_does_not_fall_back_to_standalone():
    with pytest.raises(ValueError, match="Unsupported S4 implementation"):
        small_model(implementation="standalone")


def test_parameter_counts_match_modular_recipe():
    model = make_s4("embeddings", num_numeric_features=28, num_states=8,
                    num_commodities=93, num_flows=2)
    assert model.get_num_params() == 180326
    assert model.architecture_provenance()["architecture_versions"]["backbone"] == IMPLEMENTATION


def test_changed_s4_contracts_cannot_relabel_standalone_runs():
    cfg = load_config(str(ROOT / "config/base.yaml"))
    cfg = load_config(str(ROOT / "config/models/s4.yaml"), cfg)
    aggregate = build_run_contract(cfg, model="s4", variant="embeddings", seed=26,
                                   aggregate=True, source_fingerprint="a"*64, source_domain={"test": True})
    axial = expected_contract(ROOT, Exp2Run("s4", "cross_attention_2d", 26))
    sweep = exp3_contract("s4", 36, 12, 26)
    for contract in (aggregate, axial, sweep):
        assert contract["model_config"]["implementation"] == IMPLEMENTATION
        assert contract["model_config"]["n_ssm"] == 1
        assert contract["model_config"]["tie_dropout"] is True
        assert contract["model_config"]["dt_transform"] == "softplus"
    # Old YAML without the new field still binds to the actual current backend.
    cfg["model"].pop("implementation")
    current = build_run_contract(cfg, model="s4", variant="embeddings", seed=26,
                                 aggregate=True, source_fingerprint="a"*64, source_domain={"test": True})
    assert current["model_config"]["implementation"] == IMPLEMENTATION
    identity = dict(model="s4", variant="embeddings", seed=26, aggregate=True,
                    source_fingerprint="a"*64, source_domain={"test": True})
    validate_run_contract(current, cfg, **identity)
    legacy = {**current, "model_config": dict(current["model_config"])}
    for field in ("implementation", "n_ssm", "dt_transform", "tie_dropout"):
        legacy["model_config"].pop(field)
    with pytest.raises(ValueError, match="model_config"):
        validate_run_contract(legacy, cfg, **identity)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_numerical_parity_with_normal_external_imports(tmp_path, bidirectional):
    model = small_model(bidirectional=bidirectional).eval()
    x = torch.randn(2, 7, 16, requires_grad=True)
    y, _ = model.backbone(x)
    y.square().sum().backward()
    input_path, output_path = tmp_path / "input.pt", tmp_path / "output.pt"
    torch.save({"state": model.backbone.state_dict(), "x": x.detach(),
                "config": backbone_config(d_model=16, n_layers=2, d_state=8, dropout=0.0,
                                          bidirectional=bidirectional)}, input_path)
    code = '''
import sys,torch
sys.path.insert(0,'.')
from src import utils
import src.utils.registry as registry
torch.set_num_threads(1)
p=torch.load(sys.argv[1],weights_only=True)
m=utils.instantiate(registry.model,p['config']).eval()
m.load_state_dict(p['state'])
x=p['x'].requires_grad_();y,_=m(x);y.square().sum().backward()
torch.save({'y':y.detach(),'x_grad':x.grad,
            'grads':{n:p.grad for n,p in m.named_parameters()}},sys.argv[2])
'''
    proc = subprocess.run([sys.executable, "-c", code, str(input_path), str(output_path)],
                          cwd=ROOT / "external/s4", capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    reference = torch.load(output_path, weights_only=True)
    torch.testing.assert_close(y, reference["y"], rtol=0, atol=0)
    torch.testing.assert_close(x.grad, reference["x_grad"], rtol=0, atol=0)
    for name, parameter in model.backbone.named_parameters():
        torch.testing.assert_close(parameter.grad, reference["grads"][name], rtol=0, atol=0)


def test_vendor_source_matches_disclosed_audit():
    spec = importlib.util.spec_from_file_location("s4_audit", ROOT / "scripts/check_s4_upstream.py")
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    report = audit.verify(ROOT / "external/s4")
    assert not report["errors"], json.dumps(report["errors"])
    assert report["unmodified_s4_core_files"] == 10
    assert report["files"]["modified"] == 12  # Do not misrepresent the whole tree as pristine.


@pytest.mark.parametrize("model_name,window,lag", sorted({(model, w, lag) for model, w, lag, seed in coordinates()
                                               if model in {"s4", "s4_bidir"}}))
def test_every_s4_sweep_shape_with_full_canonical_backbone(model_name, window, lag):
    cfg = resolved_config(model_name, window, lag, 26)["model"]
    settings = {key: cfg[key] for key in ("d_model", "n_layers", "d_state", "dropout",
                                        "n_ssm", "dt_transform", "tie_dropout", "implementation", "bidirectional")}
    torch.manual_seed(26)
    model = make_s4("embeddings", num_numeric_features=2*lag+4,
                    num_states=8, num_commodities=93, num_flows=2, **settings)
    assert len(model.backbone.layers) == 4
    assert model.backbone.d_model == 128
    x = torch.randn(2, window, 2*lag+4, requires_grad=True)
    result = model(x, torch.tensor([0, 7]), torch.tensor([0, 92]), torch.tensor([0, 1]))
    assert result.shape == (2, 2) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all("lr" not in getattr(p, "_optim", {}) for p in model.parameters())


@pytest.mark.parametrize("bidirectional", [False, True])
def test_modular_s4_causal_prefix_and_seed_reproducibility(bidirectional):
    torch.manual_seed(26)
    model = small_model(bidirectional=bidirectional).eval()
    torch.manual_seed(26)
    repeat = small_model(bidirectional=bidirectional).eval()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, repeat.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 12, 16)
    changed = x.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 10
    with torch.no_grad():
        before = model.backbone(x)[0]
        after = model.backbone(changed)[0]
    if bidirectional:
        assert not torch.allclose(before[:, :6], after[:, :6], rtol=1e-4, atol=1e-5)
    else:
        torch.testing.assert_close(before[:, :6], after[:, :6], rtol=1e-4, atol=1e-5)


def test_both_directions_construct_in_a_fresh_native_process():
    code = '''
from src.models import create_model
for name, direction in [('s4', False), ('s4_bidir', True)]:
    model = create_model(name, variant='embeddings', num_numeric_features=4,
                         num_states=2, num_commodities=2, num_flows=2,
                         d_model=8, n_layers=1, d_state=4)
    assert model.bidirectional is direction
    assert model.backbone.layers[0].layer.layer.bidirectional is direction
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_direction_identity_cannot_be_overridden_or_relabelled():
    from src.models.s4 import make_s4_bidir
    with pytest.raises(ValueError, match="requires bidirectional"):
        make_s4_bidir("embeddings", bidirectional=False)
    cfg = resolved_config("s4_bidir", 36, 12, 26)
    causal = exp3_contract("s4", 36, 12, 26)
    current = exp3_contract("s4_bidir", 36, 12, 26)
    identity = {**current["identity"], "source_fingerprint": current["source_fingerprint"],
                "source_domain": current["source_domain"]}
    validate_run_contract(current, cfg, **identity)
    with pytest.raises(ValueError, match="identity"):
        validate_run_contract(causal, cfg, **identity)
    relabelled = copy.deepcopy(causal)
    relabelled["identity"] = current["identity"]
    with pytest.raises(ValueError, match="model_config"):
        validate_run_contract(relabelled, cfg, **identity)


def test_modular_s4_optimizer_and_rng_resume_matches_uninterrupted_training():
    torch.manual_seed(26)
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x, ids = torch.randn(2, 7, 4), torch.tensor([0, 1])

    def step(network, optim):
        optim.zero_grad(set_to_none=True)
        loss = network(x, ids, ids, ids).square().mean()
        loss.backward()
        optim.step()
        return loss.detach()

    step(model, optimizer)
    checkpoint = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "rng": torch.get_rng_state()})
    expected_loss = step(model, optimizer)
    resumed = small_model()
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    actual_loss = step(resumed, resumed_optimizer)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def test_source_audit_rejects_modified_missing_and_added_code(tmp_path):
    spec = importlib.util.spec_from_file_location("s4_audit", ROOT / "scripts/check_s4_upstream.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    source = tmp_path / "src/core.py"
    source.parent.mkdir()
    original = b"# pinned core\n"
    source.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    (tmp_path / "UPSTREAM.json").write_text(json.dumps({
        "repository": "fixture", "commit": "fixture", "unmodified_s4_core": ["src/core.py"],
        "files": {"src/core.py": {"status": "identical", "local_sha256": digest,
                                  "upstream_sha256": digest}},
    }))
    assert not audit.verify(tmp_path)["errors"]
    source.write_text("# changed\n")
    assert audit.verify(tmp_path)["errors"]
    source.unlink()
    assert audit.verify(tmp_path)["errors"]
    source.write_bytes(original)
    added = source.parent / "__init__.py"
    added.write_text("# new import behavior\n")
    assert any("unaudited" in error for error in audit.verify(tmp_path)["errors"])
