"""Completion evidence bound to the exact selected checkpoint and run contract.

Early stopping is a protocol criterion, not proof of mathematical convergence.
Epoch-cap runs can be inspected but are not accepted by the Exp 3 paper gate.
"""
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def completion_path(checkpoint):
    return Path(checkpoint).parent / 'artifacts/training_completion.json'


def clean_contract(contract):
    if not isinstance(contract, Mapping) or any(not isinstance(k, str) for k in contract):
        raise ValueError('invalid training completion run contract')
    return {k: v for k, v in contract.items() if not k.startswith('_')}


def write_completion(checkpoint, contract, *, epochs_run, best_epoch, best_val,
                     stopped_early, patience_counter):
    # A resumed run must still possess the selected best checkpoint, not just
    # a last-epoch state. Bind the terminal summary to its embedded selection.
    import torch
    selected = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (selected.get('run_contract') != clean_contract(contract)
            or selected.get('epoch') != best_epoch - 1
            or selected.get('score') != best_val):
        raise ValueError('best checkpoint does not match the completed training selection')
    receipt = {
        'version': 1, 'state': 'finished',
        'stop_reason': 'early_stopping' if stopped_early else 'epoch_limit',
        'epochs_run': epochs_run, 'best_epoch': best_epoch, 'best_val': best_val,
        'patience_counter': patience_counter,
        'run_contract': clean_contract(contract),
        'checkpoint_sha256': checkpoint_digest(checkpoint),
    }
    path = completion_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def validate_completion(receipt, contract, checkpoint=None, *, require_early_stopping=True):
    if (not isinstance(receipt, dict) or type(receipt.get('version')) is not int
            or receipt.get('version') != 1):
        raise ValueError('missing or unsupported training completion evidence; retrain')
    reason = receipt.get('stop_reason')
    if (receipt.get('state') != 'finished'
            or reason not in {'early_stopping', 'epoch_limit'}
            or (require_early_stopping and reason != 'early_stopping')):
        raise ValueError('training did not finish by early stopping (interrupted or epoch-capped)')
    if receipt.get('run_contract') != clean_contract(contract):
        raise ValueError('training completion belongs to a different run contract')
    training = contract.get('training_config', {})
    if not isinstance(training, Mapping):
        raise ValueError('invalid completion training configuration')
    budget = training.get('epochs')
    patience = training.get('patience')
    if (type(budget) is not int or budget <= 0
            or type(patience) is not int or patience <= 0):
        raise ValueError('invalid completion epoch budget or patience')
    epochs, best, counter = (receipt.get(k) for k in
                              ('epochs_run', 'best_epoch', 'patience_counter'))
    if any(type(v) is not int for v in (epochs, best, counter)):
        raise ValueError('invalid completion epoch/counter evidence')
    if not 1 <= best <= epochs <= budget:
        raise ValueError('completion epochs are outside the training budget')
    if reason == 'early_stopping' and (patience <= 0 or counter < patience or counter >= epochs):
        raise ValueError('completion does not satisfy early-stopping patience')
    if reason == 'epoch_limit' and (epochs != training.get('epochs') or not 0 <= counter < epochs):
        raise ValueError('completion does not satisfy the configured epoch limit')
    value = receipt.get('best_val')
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid completion validation MSE')
    digest = receipt.get('checkpoint_sha256')
    if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise ValueError('missing completion checkpoint digest')
    if checkpoint is not None and checkpoint_digest(checkpoint) != digest:
        raise ValueError('selected checkpoint differs from completed training')


def read_completion(checkpoint, contract, *, require_early_stopping=True):
    path = completion_path(checkpoint)
    if not path.is_file():
        raise ValueError(f'No training completion evidence at {path}; retrain')
    receipt = json.loads(path.read_text())
    validate_completion(receipt, contract, checkpoint, require_early_stopping=require_early_stopping)
    return receipt


def validate_completion_artifact(payload, metrics_path):
    checkpoint = Path(metrics_path).parent / 'best.pth'
    receipt = read_completion(checkpoint, payload['run_contract'])
    if receipt != payload.get('training_completion'):
        raise ValueError('metrics completion evidence differs from training artifact')
