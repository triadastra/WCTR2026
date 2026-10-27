#!/usr/bin/env python3
"""Prepare and run the canonical Exp 3 sweep without launching Exp 1/2."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.exp3_protocol import coordinates, manifest, resolved_config, run_name


def preflight(data_dir):
    from src.data.dataloader import DataConfig, TradeDataPipeline
    from src.utils.config import get_data_config, resolve_data_paths
    cfg = resolved_config(*coordinates()[0])
    data = get_data_config(cfg)
    data = resolve_data_paths(data, data_dir)
    pipe = TradeDataPipeline(DataConfig(**data))
    pipe.load_data()  # source bytes, population, calendar, duplicate-key checks
    from src.utils.experiment_manifest import exp1_source_domain
    if pipe.source_domain != exp1_source_domain(manifest()):
        raise ValueError('Source domain/audit differs from the canonical manifest')
    pipe.create_splits()  # train-only cohort and scaling
    print(f'PASS: source verified, {len(pipe.kept_combos)} training-selected series')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['plan', 'check', 'train', 'collect'], nargs='?', default='check')
    ap.add_argument('--data-dir', type=Path, default=ROOT)
    args = ap.parse_args()
    os.chdir(ROOT)
    m = manifest()
    grid = coordinates()
    if args.action == 'plan':
        print(json.dumps({'runs': len(grid), 'models': m['PAPER_SWEEP_MODELS'],
                          'windows': m['PAPER_WINDOWS'], 'lags': m['PAPER_LAGS'],
                          'seeds': m['PAPER_SEEDS'], 'batch_size': int(m['PAPER_SWEEP_BATCH']),
                          'train_target_start': m['PAPER_SWEEP_TRAIN_START'],
                          'selection': '2024 validation MSE only',
                          'test': 'Jan-May 2025 rolling one-month-ahead; descriptive sensitivity only',
                          'execution': 'sequential on one CUDA device'}, indent=2))
        return 0
    if args.action == 'collect':
        from scripts.build_paper_results import _sweep_rows
        import pandas as pd
        rows = _sweep_rows(ROOT / 'outputs/sweep')
        if not rows:
            raise ValueError('No complete validated sweep')
        path = ROOT / 'outputs/sweep/exp3_cells.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        print(path)
        return 0
    preflight(args.data_dir.resolve())
    from src.utils.runtime_readiness import check_runtime
    runtime = check_runtime(ROOT, include_exp2=False)
    if not runtime['passed']:
        raise RuntimeError('Exp3 runtime preflight failed: ' + '; '.join(runtime['errors']))
    if args.action == 'check':
        print('PASS: Exp3 imports and CUDA execution verified')
        return 0
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Canonical training requires CUDA; preflight only on this host')
    lock = ROOT / 'outputs/.run_all_experiments.lock'
    lock.parent.mkdir(exist_ok=True)
    lock.mkdir()  # fail closed on concurrent invocation or an uncleared interrupted run
    try:
        for model, window, lag, seed in grid:
            out = ROOT / 'outputs/sweep' / run_name(model, window, lag, seed)
            common = ['--exp3', '--model', model, '--variant', 'embeddings', '--seed', str(seed),
                      '--input-len', str(window), '--lag-count', str(lag),
                      '--train-target-start', m['PAPER_SWEEP_TRAIN_START'],
                      '--data-dir', str(args.data_dir.resolve()), '--out-dir', str(out)]
            gate = [sys.executable, 'scripts/run_is_current.py', str(out / 'metrics.json'),
                    '--exp3', '--expected-model', model, '--expected-variant', 'embeddings',
                    '--expected-seed', str(seed), '--input-len', str(window), '--lag-count', str(lag)]
            if subprocess.run(gate).returncode == 0:
                continue
            out.mkdir(parents=True, exist_ok=True)
            (out / 'metrics.json').unlink(missing_ok=True)
            for script, extra, log in [
                ('train.py', ['--batch-size', m['PAPER_SWEEP_BATCH']], 'train.log'),
                ('inference.py', ['--normalized', '--device', 'cuda', '--checkpoint', str(out / 'best.pth')], 'inference.log'),
            ]:
                print(f'{script}: {out.name}', flush=True)
                with (out / log).open('w') as stream:
                    subprocess.run([sys.executable, f'scripts/{script}', *common, *extra],
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
            subprocess.run(gate, check=True)
    finally:
        lock.rmdir()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'[Exp3 FAIL] {exc}', file=sys.stderr)
        raise SystemExit(1)
