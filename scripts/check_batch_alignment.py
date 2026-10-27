#!/usr/bin/env python3
"""Verify native epoch steps on the actual monthly release without training."""
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataloader import DataConfig, TradeDataPipeline
from src.utils.config import get_data_config, load_config
from src.utils.exp3_protocol import coordinates, resolved_config


def main():
    base = load_config(str(ROOT / 'config/base.yaml'))
    source = TradeDataPipeline(DataConfig(**get_data_config(base)))
    source.load_data()

    def prepare(cfg):
        pipe = TradeDataPipeline(DataConfig(**get_data_config(cfg)))
        pipe.df_full = source.df_full.copy()
        pipe.create_splits()
        return pipe

    aggregate = load_config(str(ROOT / 'config/exp1.yaml'), base)
    aggregate['data']['aggregate'] = True
    pipe = prepare(aggregate)
    tr = pipe.get_dataloaders(aggregate['training']['batch_size'], num_workers=0)[0]
    assert len(tr) == 4 and len(tr.dataset) == 132 and not tr.drop_last
    assert {len(b) for b in tr.batch_sampler} == {33}
    print(json.dumps({'experiment': 1, 'neural_steps_per_epoch': len(tr), 'batch_size': tr.batch_size}), flush=True)
    del pipe, tr

    pipe = prepare(base)
    flat = pipe.get_dataloaders(base['training']['batch_size'], num_workers=0)[0]
    combo = pipe.get_dataloaders(1, num_workers=0, combo=True)[0]
    assert len(flat) == len(combo) == 132
    assert len(flat.dataset) == len(combo.dataset) * len(pipe.kept_combos)
    assert {len(b) for b in flat.batch_sampler} == {918}
    print(json.dumps({'experiment': 2, 'neural_steps_per_epoch': len(flat), 'flat_batch_size': 918, 'combo_batch_size': 1}), flush=True)
    del pipe, flat, combo
    gc.collect()

    seen = set()
    for model, window, lag, seed in coordinates():
        if (window, lag) in seen:
            continue  # All three model families/seeds share this data contract.
        seen.add((window, lag))
        cfg = resolved_config(model, window, lag, seed)
        pipe = prepare(cfg)
        tr = pipe.get_dataloaders(cfg['training']['batch_size'], num_workers=0)[0]
        months = {s['target_time'] for s in tr.dataset.samples}
        assert len(tr) == len(months) == 108
        assert len(tr.dataset) == 108 * len(pipe.kept_combos)
        assert {len(b) for b in tr.batch_sampler} == {918}
        assert sum(len(b) for b in tr.batch_sampler) == len(tr.dataset)
        print(json.dumps({'experiment': 3, 'window': window, 'lag': lag, 'neural_steps_per_epoch': len(tr), 'batch_size': 918}), flush=True)
        del pipe, tr, months
        gc.collect()
    print('All native epoch-step checks passed', flush=True)


if __name__ == '__main__':
    main()
