"""Exp 3 must reject relabelled or differently trained benchmark cells."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from src.utils.exp3_protocol import coordinates, expected_contract, validate_contract


def payload():
    contract = expected_contract('lstm', 6, 6, 26)
    return {'model': 'lstm', 'variant': 'embeddings', 'seed': 26,
            'run_contract': contract,
            'training_completion': {
                'version': 1, 'state': 'finished', 'stop_reason': 'early_stopping',
                'epochs_run': 60, 'best_epoch': 10, 'best_val': 0.01,
                'patience_counter': 50, 'run_contract': deepcopy(contract),
                'checkpoint_sha256': 'a' * 64,
            }}


def test_exact_grid_and_valid_contract():
    assert len(coordinates()) == len(set(coordinates())) == 192
    assert validate_contract(payload(), 'lstm', 6, 6, 26)[0]


@pytest.mark.parametrize('section,key,value', [
    ('training_config', 'batch_size', 903),
    ('training_config', 'epochs', 1),
    ('training_config', 'lr', 0.01),
    ('selection', 'monitor', 'test_loss'),
    ('data_config', 'cohort_from_train_only', False),
    ('data_config', 'train_target_start', '2010-01-01'),
    ('data_config', 'target_measure', 'invalid_measure'),
    ('model_config', 'num_layers', 1),
])
def test_off_protocol_run_cannot_be_reused(section, key, value):
    p = deepcopy(payload())
    p['run_contract'][section][key] = value
    assert not validate_contract(p, 'lstm', 6, 6, 26)[0]


def test_relabelled_seed_and_out_of_grid_coordinates_rejected():
    assert not validate_contract(payload(), 'lstm', 6, 6, 105)[0]
    assert not validate_contract(payload(), 'lstm', 7, 6, 26)[0]


def test_heatmap_uses_publication_gate(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'util/build_sweep_heatmaps.py'
    spec = importlib.util.spec_from_file_location('heatmaps', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'RUNS', tmp_path)
    from scripts import build_paper_results
    def reject(_):
        raise RuntimeError('stale canonical run')
    monkeypatch.setattr(build_paper_results, '_sweep_rows', reject)
    with pytest.raises(RuntimeError, match='stale canonical'):
        module.load_cells()


@pytest.mark.parametrize('model', ['lstm', 'transformer', 's4', 's4_bidir'])
@pytest.mark.parametrize('window,lag', [(6, 6), (36, 36)])
def test_sweep_models_forward_backward(model, window, lag):
    import torch
    from src.models import create_model
    from src.utils.exp3_protocol import resolved_config
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = deepcopy(resolved_config(model, window, lag, 26)['model'])
        cfg.pop('name', None)
        cfg.pop('variant', None)
        dims = cfg.pop('embedding_dims')
        net = create_model(name=model, variant='embeddings',
                           num_numeric_features=4 + 2 * lag, num_states=8,
                           num_commodities=97, num_flows=2,
                           state_embed_dim=dims['state'], comm_embed_dim=dims['commodity'],
                           flow_embed_dim=dims['flow'], **cfg)
        x = torch.rand(2, window, 4 + 2 * lag)
        ids = torch.zeros(2, dtype=torch.long)
        prediction = net(x, ids, ids, ids)
        assert prediction.shape == (2, 2)
        assert torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    finally:
        torch.set_num_threads(previous)


def test_standalone_training_failure_suppresses_inference(tmp_path, monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace
    import torch
    from scripts import run_exp3
    from src.utils import runtime_readiness
    monkeypatch.setattr(runtime_readiness, 'check_runtime', lambda *args, **kwargs: {'passed': True})
    monkeypatch.setattr(run_exp3, 'ROOT', tmp_path)
    monkeypatch.setattr(run_exp3, 'preflight', lambda _: None)
    monkeypatch.setattr(run_exp3, 'coordinates', lambda: [('lstm', 6, 6, 26)])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(sys, 'argv', ['run_exp3.py', 'train', '--data-dir', str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if 'scripts/train.py' in command:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(run_exp3.subprocess, 'run', fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        run_exp3.main()
    assert any('scripts/train.py' in command for command in calls)
    assert not any('scripts/inference.py' in command for command in calls)
    assert not (tmp_path / 'outputs/.run_all_experiments.lock').exists()
