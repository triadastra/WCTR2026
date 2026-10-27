"""Reporting uses complete source-validated populations and one inference unit."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from util import make_significance as sig
from util import make_cd_diagram as cd
from util import make_axial_vs_flat as axial
from util import make_vs_baseline as baseline
from util import make_paper_tables as paper


@pytest.fixture
def losses():
    # Unequal lane counts distinguish pooled loss from equally weighted states.
    keys = [(f'S{s}', f'C{c}', 'Import', f'2025-{m:02d}-01')
            for s in range(8) for c in range(s + 1) for m in range(1, 6)]
    index = pd.MultiIndex.from_tuples(keys, names=sig.KEY)
    arms = list(baseline.BASELINES)
    arms += [f'{b}_{v}' for b in axial.BACKBONES for v in
             ('onehot', 'embeddings', 'cross_attention_2d', 'cross_attention_3d', 'cross_attention_4d')]
    arms += ['s4nd_2d', 's4nd_2d_state']
    return {arm: (pd.Series(np.arange(len(index)) / 100 + 1 + i / 10, index=index),
                  [1 + i / 10] * 3) for i, arm in enumerate(arms)}


def test_statistical_loader_invokes_full_validation(monkeypatch):
    def refuse(*args):
        raise RuntimeError('inexact canonical roster')
    monkeypatch.setattr(sig, 'validated_exp2_artifacts', refuse)
    with pytest.raises(RuntimeError, match='canonical roster'):
        sig.load_model_losses()


def test_loader_uses_payload_identity_and_all_seed_populations(tmp_path, monkeypatch):
    artifacts = []
    for seed in sig.SEEDS:
        path = tmp_path / f'arbitrary_{seed}'
        path.mkdir()
        frame = pd.DataFrame({'State': ['A', 'B'], 'Commodity': ['C', 'C'],
                              'Import/Export': ['Import'] * 2, 'Time': ['2025-01-01'] * 2,
                              'true_value': [0., 0.], 'true_weight': [0., 0.],
                              'pred_value': [2., 2.], 'pred_weight': [2., 2.]})
        frame.iloc[::-1].to_csv(path / 'predictions.csv', index=False)
        artifacts.append((SimpleNamespace(model='lstm', variant='embeddings', seed=seed),
                          path / 'metrics.json', {'predictions_file': 'predictions.csv'}))
    monkeypatch.setattr(sig, 'validated_exp2_artifacts', lambda *args: artifacts)
    assert sig.load_model_losses()['lstm_embeddings'][1] == [4.] * 3
    assert sig.load_model_losses(metric='mae')['lstm_embeddings'][1] == [2.] * 3
    # Explicit metric selection does not mutate subsequent MSE analyses.
    assert sig.load_model_losses()['lstm_embeddings'][1] == [4.] * 3
    frame.iloc[:1].to_csv(artifacts[-1][1].parent / 'predictions.csv', index=False)
    with pytest.raises(RuntimeError, match='population differs'):
        sig.load_model_losses()
    artifacts.pop()
    with pytest.raises(RuntimeError, match='seed roster'):
        sig.load_model_losses()


def test_significance_and_cd_test_eight_states(losses, monkeypatch, tmp_path):
    with pytest.warns(RuntimeWarning):
        table = sig.significance_table(losses)
    assert table.n_blocks.eq(8).all()
    assert table.inference_unit.eq('state').all()
    assert 'tied_with_best' not in table
    seen = {}
    original = cd.friedman_nemenyi
    def capture(matrix):
        seen['matrix'] = matrix
        return original(matrix)
    monkeypatch.setattr(cd, 'load_model_losses', lambda: losses)
    monkeypatch.setattr(cd, 'friedman_nemenyi', capture)
    monkeypatch.setattr(cd, 'make_figure', lambda *args: None)
    monkeypatch.setattr(cd, 'OUTDIR', tmp_path)
    with pytest.warns(RuntimeWarning):
        assert cd.main() == 0
    assert seen['matrix'].shape == (8, len(losses))
    np.testing.assert_allclose(seen['matrix'], sig.state_loss_matrix(losses))


def test_axial_and_baseline_use_same_observation_weights_and_state_clusters(losses, monkeypatch, tmp_path):
    monkeypatch.setattr(axial, 'load_model_losses', lambda: losses)
    monkeypatch.setattr(axial, 'OUTDIR', tmp_path)
    assert axial.main() == 0
    table = pd.read_csv(tmp_path / 'axial_vs_flat.csv')
    assert len(table) == 36
    assert table.n_blocks.eq(8).all()
    table = baseline.baseline_table(losses)
    assert table.n_blocks.eq(8).all()
    assert table.inference_unit.eq('series_time').all()
    assert table.cluster_unit.eq('state').all()


def test_mismatched_states_fail_instead_of_intersection(losses):
    arm = 'lstm_onehot'
    loss, seeds = losses[arm]
    losses[arm] = (loss[loss.index.get_level_values(0) != 'S0'], seeds)
    with pytest.raises(RuntimeError, match='populations differ'):
        sig.state_loss_matrix(losses)
    with pytest.raises(RuntimeError, match='populations differ'):
        baseline.baseline_table(losses)


def test_primary_direction_matches_pooled_loss_with_state_sensitivity(losses):
    base = losses[baseline.BASELINE][0]
    # Seven small states slightly better; largest state much worse: pooled and macro differ.
    state = base.index.get_level_values(0)
    difference = np.where(state == 'S7', -6., 1.)
    losses['lstm_onehot'] = (base - difference, [0.] * 3)
    row = baseline.baseline_table(losses).set_index('arm').loc['lstm_onehot']
    assert row.mean_state_delta > 0
    assert row.mean_lane_delta < 0
    assert not row.beats_baseline
    assert not row.lower_pooled_loss
    assert row.improvement_pct < 0


def test_separate_s4nd_layout_cells():
    rows = [dict(model='s4nd', variant=v, improvement_pct=i, mean_metric=1.,
                 sig_raw=False, sig_fdr=False) for i, v in enumerate(['2d', '2d_state'])]
    _, _, matrix, *_ = baseline._grid_from_rows(rows)
    assert np.isfinite(matrix).sum() == 2
    assert paper.STRUCT['2d'] != paper.STRUCT['2d_state']
    with pytest.raises(RuntimeError, match='Multiple arms'):
        baseline._grid_from_rows(rows + rows[:1])


def test_plot_only_cannot_reuse_cached_statistics(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, 'OUTDIR', tmp_path)
    monkeypatch.setattr(baseline.sys, 'argv', ['make_vs_baseline.py', '--plot-only'])
    def refuse():
        raise RuntimeError('source validation failed')
    monkeypatch.setattr(baseline, 'load_model_losses', refuse)
    assert baseline.main() == 2


def test_primary_figures_render_the_new_clustered_columns(losses, monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, 'OUTDIR', tmp_path)
    table = baseline.baseline_table(losses)
    baseline.make_heatmap(table, float(losses[baseline.BASELINE][0].mean()))
    import make_significance as ms
    monkeypatch.setattr(ms, 'load_model_losses', lambda metric='mse': losses)
    assert baseline.pair_mode() == 0
    assert (tmp_path / 'vs_baseline_pair.png').stat().st_size > 1000
