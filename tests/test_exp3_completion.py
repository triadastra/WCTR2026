"""Exercise completion evidence through real training and checkpoint files."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from src.training.trainer import Trainer
from src.utils.training_completion import (
    completion_path, read_completion, validate_completion, validate_completion_artifact,
)
from tests.test_exp1_training_integrity import _ZeroModel, _batch, _trainer_config
from tests.test_exp3_integrity import payload
from src.utils.exp3_protocol import validate_contract, resolved_config


def trainer(tmp_path, epochs=5):
    cfg = _trainer_config(tmp_path)
    cfg['training'].update(epochs=epochs, patience=2)
    contract = {'training_config': deepcopy(cfg['training'])}
    cfg['logging']['run_contract'] = contract
    return Trainer(cfg), contract


def fit(t):
    return t.fit(_ZeroModel(), [_batch([1, 2])], [_batch([1, 2])])


def test_successful_early_stop_binds_checkpoint_and_metrics(tmp_path):
    t, contract = trainer(tmp_path)
    fit(t)
    ckpt = tmp_path / 'checkpoints/best.pth'
    receipt = read_completion(ckpt, contract)
    assert receipt['epochs_run'] == 3
    assert receipt['stop_reason'] == 'early_stopping'
    artifact = {'run_contract': contract, 'training_completion': receipt}
    validate_completion_artifact(artifact, ckpt.parent / 'metrics.json')
    artifact['training_completion'] = dict(receipt, best_val=99)
    with pytest.raises(ValueError, match='differs'):
        validate_completion_artifact(artifact, ckpt.parent / 'metrics.json')
    ckpt.write_bytes(ckpt.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='checkpoint differs'):
        read_completion(ckpt, contract)


def test_epoch_cap_cannot_be_published(tmp_path):
    t, contract = trainer(tmp_path, epochs=1)
    fit(t)
    with pytest.raises(ValueError, match='epoch-capped'):
        read_completion(tmp_path / 'checkpoints/best.pth', contract)


def test_restarted_interrupted_training_clears_old_completion(tmp_path, monkeypatch):
    t, contract = trainer(tmp_path)
    fit(t)
    ckpt = tmp_path / 'checkpoints/best.pth'
    assert completion_path(ckpt).exists()
    t, _ = trainer(tmp_path)
    monkeypatch.setattr(t, 'train_epoch', lambda: (_ for _ in ()).throw(RuntimeError('interrupted')))
    with pytest.raises(RuntimeError, match='interrupted'):
        fit(t)
    assert not completion_path(ckpt).exists()
    with pytest.raises(ValueError, match='No training completion'):
        read_completion(ckpt, contract)
    t.logger.close(state='failed')


@pytest.mark.parametrize('field,value', [
    ('state', 'running'), ('stop_reason', 'epoch_limit'), ('epochs_run', 5001),
    ('best_epoch', 61), ('best_val', float('nan')), ('patience_counter', 49),
])
def test_exp3_gate_rejects_bad_completion(field, value):
    p = payload()
    p['training_completion'][field] = value
    assert not validate_contract(p, 'lstm', 6, 6, 26)[0]


def test_missing_completion_rejected():
    p = payload()
    p.pop('training_completion')
    assert not validate_contract(p, 'lstm', 6, 6, 26)[0]


def test_exp3_linear_head_keeps_attention_heads_and_old_default():
    from src.models import create_model
    cfg = deepcopy(resolved_config('transformer', 6, 6, 26)['model'])
    assert cfg.pop('prediction_head') == 'linear'
    cfg.pop('name'); cfg.pop('variant'); cfg.pop('embedding_dims')
    kwargs = dict(name='transformer', variant='embeddings', num_numeric_features=16,
                  num_states=8, num_commodities=97, num_flows=2, **cfg)
    old = create_model(**kwargs)
    new = create_model(**kwargs, prediction_head='linear')
    assert isinstance(old.output_layer, nn.Sequential)
    assert isinstance(new.output_layer, nn.Linear)
    assert (new.output_layer.in_features, new.output_layer.out_features) == (128, 2)
    assert new.num_heads == 8
    assert sum(p.numel() for p in old.parameters()) - sum(p.numel() for p in new.parameters()) == 8128


def test_collector_rejects_missing_completion_before_publishing(tmp_path, monkeypatch):
    import json
    from scripts import build_paper_results as collector
    p = payload()
    p.pop('training_completion')
    run = tmp_path / 'lstm_embeddings_w6_l6_seed26'
    run.mkdir()
    (run / 'metrics.json').write_text(json.dumps(p))
    monkeypatch.setattr(collector, 'corrected_per_combo_artifact', lambda _: (True, 'ok'))
    monkeypatch.setattr(collector, 'corrected_prediction_artifact', lambda *_, **kwargs: (True, 'ok'))
    with pytest.raises(RuntimeError, match='missing or unsupported training completion'):
        collector._sweep_rows(tmp_path)
