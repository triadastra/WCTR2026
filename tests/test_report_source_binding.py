"""Legacy report entry points cannot bypass the current artifact gates."""
import importlib
import pandas as pd
import pytest
from util import make_paper_figures as figures
from util import make_rank_slopegraph as slope

@pytest.mark.parametrize('module,loader,collector', [
    (figures, 'load_exp2', '_exp2_rows'), (figures, 'load_exp1_mean', '_exp1_rows'),
    (slope, 'load', '_exp2_rows')])
def test_figures_require_live_validation(module, loader, collector, monkeypatch):
    def refuse(*args):
        raise RuntimeError('source mismatch')
    monkeypatch.setattr(module, collector, refuse)
    with pytest.raises(RuntimeError, match='source mismatch'):
        getattr(module, loader)()

@pytest.mark.parametrize('name', ['refresh_exp2_sheets', 'consolidate_workbook'])
def test_in_place_workbook_updaters_cannot_publish(name):
    assert importlib.import_module('util.' + name).main() == 2


def test_figure_uses_current_s4_parameter_count():
    assert figures.get_params('s4', 'embeddings', 123456) == 123456
    assert figures.get_params('s4', 'embeddings', None) == 0


def test_rank_data_reflects_reversed_run_results(monkeypatch):
    rows = [dict(model=model, variant='embeddings', seed=seed,
                 norm_mse_combined=mse, mae=mae, mase=mae)
            for model, mse, mae in [('lstm', 1., 2.), ('gru', 2., 1.)]
            for seed in [26, 105, 115]]
    monkeypatch.setattr(slope, '_exp2_rows', lambda _: rows)
    before = slope.load().set_index('model')
    assert before.loc['lstm','r_mse'] == 1
    for row in rows:
        row['norm_mse_combined'] = 3 - row['norm_mse_combined']
    after = slope.load().set_index('model')
    assert after.loc['lstm','r_mse'] == 2
