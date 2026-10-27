"""Frozen monthly inputs must preserve targets and fail closed on corruption."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.dataloader import DataConfig, TradeDataPipeline, KEYS
from src.utils.config import resolve_data_paths
from src.utils.run_contract import semantic_data_config


def _write(tmp_path, frame):
    path = tmp_path / 'monthly.csv'
    frame.to_csv(path, index=False)
    files = [dict(flow='monthly', name=path.name, bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
    digest = hashlib.sha256(json.dumps(dict(version=1, files=files), sort_keys=True,
                                      separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return DataConfig(source_format='monthly', aggregated_csv=str(path),
                      target_measure='legacy_mixed', duplicate_policy='error',
                      expected_source_fingerprint=digest, start='2020-01-01', end='2020-03-01',
                      train_end='2020-01-01', val_start='2020-02-01', val_end='2020-02-01',
                      test_start='2020-03-01', test_end='2020-03-01')


def _rows():
    return pd.DataFrame([dict(State='Arkansas', Commodity='01 Live Animals',
                             Time=f'2020-0{i}-01', Value=i*100, Weight=i*10,
                             SourceRows=2, **{'Import/Export': 'Export'})
                         for i in range(1, 4)])


def test_monthly_values_are_read_directly_and_audit_columns_are_not_features(tmp_path):
    frame = _rows()
    pipe = TradeDataPipeline(_write(tmp_path, frame))
    raw = pipe.load_data()
    grid = pipe.build_grid(raw)
    assert grid.Value.tolist() == [100, 200, 300]
    assert 'SourceRows' not in raw and 'Country' not in raw
    assert pipe.duplicate_audit['groups'] == 0
    assert pipe.source_files[0]['flow'] == 'monthly'
    with pytest.raises(ValueError, match='Duplicate monthly'):
        pipe.build_grid(pd.concat([raw, raw.iloc[:1]]))
    with pytest.raises(ValueError, match='Incomplete preaggregated'):
        pipe.build_grid(raw.iloc[:2])


@pytest.mark.parametrize('problem,message', [
    ('duplicate', 'duplicate monthly'), ('missing', 'incomplete monthly'),
    ('blank', 'missing Value'), ('malformed', 'Malformed numeric'),
    ('date', 'YYYY-MM-01'), ('flow', 'invalid flow'), ('negative', 'Negative trade'),
])
def test_invalid_monthly_release_is_never_repaired(tmp_path, problem, message):
    frame = _rows()
    if problem == 'duplicate': frame = pd.concat([frame, frame.iloc[:1]])
    if problem == 'missing': frame = frame.drop(index=1)
    if problem == 'blank': frame.loc[1, 'Value'] = None
    if problem == 'malformed': frame['Value'] = frame.Value.astype(str); frame.loc[1, 'Value'] = 'unknown'
    if problem == 'date': frame.loc[1, 'Time'] = '2020-02-02'
    if problem == 'flow': frame.loc[1, 'Import/Export'] = '0'
    if problem == 'negative': frame.loc[1, 'Value'] = -1
    with pytest.raises(ValueError, match=message):
        TradeDataPipeline(_write(tmp_path, frame)).load_data()


def test_altered_monthly_bytes_are_rejected(tmp_path):
    cfg = _write(tmp_path, _rows())
    path = Path(cfg.aggregated_csv)
    path.write_text(path.read_text().replace(',100,', ',101,'))
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        TradeDataPipeline(cfg).load_data()


def test_monthly_path_obeys_explicit_data_root_and_is_not_semantic(tmp_path):
    data = dict(source_format='monthly', aggregated_csv='data/processed/monthly.csv')
    resolved = resolve_data_paths(data, tmp_path)
    assert resolved['aggregated_csv'] == str(tmp_path / data['aggregated_csv'])
    assert semantic_data_config({'data': data}) == semantic_data_config({'data': resolved})


def test_month_batches_match_combo_steps_even_with_larger_eval_batches():
    from tests.test_tabular_target_alignment import _pipe
    pipe = _pipe(True)
    pipe.config.batch_by_target_month = True
    flat = pipe.get_dataloaders(7, num_workers=0, shuffle_train=False)[0]
    combo, _, test = pipe.get_dataloaders(7, num_workers=0, combo=True, shuffle_train=False)
    assert len(flat) == len(combo)
    assert combo.batch_size == 1 and test.batch_size == 7
    assert sum(len(batch) for batch in flat.batch_sampler) == len(flat.dataset)
