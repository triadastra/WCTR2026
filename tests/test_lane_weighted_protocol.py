"""Author-selected weights agree across training and paired reporting."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.training.losses import MSELoss, MaskedMSELoss
from util.make_significance import KEY, paired_comparison, point_loss
from ext.eval.resampling import cluster_robust_paired_t


def test_padded_training_loss_and_gradient_weight_every_valid_observation():
    # State A has two valid lanes, B has one. The fourth slot is padding.
    prediction = torch.tensor([[[1., 3.], [1., 3.], [2., 4.], [100., 100.]]], requires_grad=True)
    target = torch.zeros((1, 4))
    mask = torch.tensor([[True, True, True, False]])
    loss = MaskedMSELoss()(prediction, target, target, mask)
    assert loss.item() == pytest.approx(20 / 3)  # (5 + 5 + 10) / 3, not (5 + 10) / 2
    assert loss.item() == pytest.approx(MSELoss()(prediction[mask], torch.zeros((3, 2))).item())
    loss.backward()
    torch.testing.assert_close(prediction.grad[0, 0], prediction.grad[0, 1])
    assert prediction.grad[0, 3].abs().sum().item() == 0


def test_primary_mean_preserves_unequal_observation_counts():
    # One lane per state but variable observed months: collapsing lanes first
    # would change the author-selected valid-observation objective.
    keys = [(f'S{s}', 'C', 'Import', f'2025-{m:02d}-01')
            for s in range(8) for m in range(1, 1 + (5 if s == 7 else 1))]
    index = pd.MultiIndex.from_tuples(keys, names=KEY)
    left = pd.Series([6. if k[0] == 'S7' else -1. for k in keys], index=index)
    right = left * 0
    result = paired_comparison(left, right)
    assert result['mean_lane_delta'] == pytest.approx(23 / 12)
    assert result['mean_state_delta'] == pytest.approx(-1 / 8)
    assert result['primary_estimand'] == 'lane_weighted'
    assert result['n_clusters'] == 8 and result['n_observations'] == 12
    assert result['n_lanes'] == 8


def test_month_replication_does_not_invent_independent_clusters():
    diff = np.array([-2., 1., 3., -1., 2., 4., -3., 2.])
    original = cluster_robust_paired_t(diff, np.arange(8))
    repeated = cluster_robust_paired_t(np.repeat(diff, 5), np.repeat(np.arange(8), 5))
    assert repeated == pytest.approx(original)


def test_point_loss_retains_equal_value_and_weight_channels():
    frame = pd.DataFrame(dict(pred_value=[1., 2.], pred_weight=[3., 4.],
                              true_value=[0., 0.], true_weight=[0., 0.]))
    np.testing.assert_allclose(point_loss(frame), [5., 10.])


def test_transformer_flat_and_random_forest_arms_have_canonical_contracts():
    from pathlib import Path
    from src.utils.exp2_protocol import expected_contract, expected_runs
    from src.utils.experiment_manifest import load_manifest
    root = Path(__file__).resolve().parents[1]
    runs = expected_runs(load_manifest(root / 'config/experiment_manifest.env'))
    # Dispatch and contract generation must agree for the arms restored from main.
    for run in runs:
        if run.seed == 26 and ((run.model == 'transformer' and run.variant in ('cross_attention', 'film_attention'))
                               or run.model == 'random_forest'):
            contract = expected_contract(root, run)
            assert contract is not None


def test_retired_dm_entrypoint_cannot_publish_independent_lane_pvalues(monkeypatch):
    from ext.eval import diebold_mariano as dm
    monkeypatch.setattr(dm, 'load_arms', lambda: pytest.fail('Retired report must not score runs'))
    assert dm.main() == 2


def test_compact_workbook_rejects_before_reading_cached_results(monkeypatch):
    from util import make_condensed as compact
    def refuse(*args):
        raise RuntimeError('completion is missing')
    monkeypatch.setattr(compact, '_exp1_rows', refuse)
    monkeypatch.setattr(compact.pd, 'ExcelFile', lambda *args: pytest.fail('Cached workbook read'))
    with pytest.raises(RuntimeError, match='completion is missing'):
        compact.main()
