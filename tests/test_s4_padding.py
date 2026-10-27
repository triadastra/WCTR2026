"""Shared occupancy validation and isolated real-S4ND integration checks."""
from pathlib import Path
from types import SimpleNamespace
import copy
import subprocess
import sys

import pytest
import torch

from src.utils.benchmark_protocol import BENCHMARK_PROTOCOL_VERSION
from src.utils.s4_benchmark import grid_cell_mask
from src.utils import s4_checkpoint_provenance
from src.utils.exp2_protocol import expected_contract, expected_runs, validate_contract
from src.utils.experiment_manifest import load_manifest


@pytest.mark.parametrize('mask', [torch.ones(2, 4), torch.full((2, 3), float('nan')),
                                  torch.full((2, 3), 0.5), torch.zeros(2, 3)])
def test_bad_occupancy_cannot_silently_enter_a_grid(mask):
    with pytest.raises(ValueError, match='mask'):
        grid_cell_mask((2, 3), SimpleNamespace(), {'valid_mask': mask}, device='cpu')


def test_missing_grid_mask_is_rejected_but_flat_models_need_no_mask():
    with pytest.raises(ValueError, match='missing'):
        grid_cell_mask((2, 3), SimpleNamespace(_name_='s4nd_2d_state'), {}, device='cpu')
    assert grid_cell_mask((2,), SimpleNamespace(_name_='s4_embeddings'), {}, device='cpu') is None


@pytest.mark.parametrize('ids', [torch.tensor(0), torch.empty(1, 0),
                                torch.tensor([[[0, 1]]]), torch.tensor([[[-1]]]),
                                torch.tensor([[[2]]]), torch.tensor([[[0.5]]])])
def test_per_flow_grid_rejects_ambiguous_or_invalid_identity(ids):
    dataset = SimpleNamespace(kept_mask=torch.ones(2, 3, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match='flow[_ ]ID|flow_ids'):
        grid_cell_mask((1, 2, 3), dataset, {'flow_ids': ids}, device='cpu')


def test_vendor_contract_uses_the_current_source_protocol():
    assert s4_checkpoint_provenance.BENCHMARK_PROTOCOL_VERSION == BENCHMARK_PROTOCOL_VERSION


@pytest.mark.parametrize('variant', ['2d_state', '2d', '3d', '4d'])
def test_result_reuse_and_collection_reject_old_padding_or_loss(variant):
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / 'config/experiment_manifest.env')
    run = next(r for r in expected_runs(manifest, [26])
               if r.model == 's4nd' and r.variant == variant)
    contract = expected_contract(root, run, manifest)
    payload = {'model': run.model, 'variant': run.variant, 'seed': run.seed,
               'run_contract': contract}
    assert validate_contract(payload, root)[0]
    assert contract['model_config']['grid_padding'] == s4_checkpoint_provenance.S4ND_PADDING_VERSION
    for drift in ('padding', 'loss'):
        old = copy.deepcopy(payload)
        if drift == 'padding':
            old['run_contract']['model_config'].pop('grid_padding')
        else:
            old['run_contract']['training_config']['loss_reduction'] = 'valid-target-mean-v1'
        assert not validate_contract(old, root)[0]


def test_real_s4nd_padding_layers_loss_gradients_and_checkpoints():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'tests/check_s4_padding.py')],
                            cwd=root / 'external/s4', capture_output=True, text=True,
                            timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ALL S4ND PADDING CHECKS PASSED' in result.stdout
