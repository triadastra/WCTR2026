"""CPU data-only check of every canonical S4 representation.

Run from external/s4: python ../../tests/check_s4_month_batches.py
No model construction, optimizer updates, or experiment outputs.
"""
import gc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path.cwd()))
from src.dataloaders.trade_unified import TradeS4Embeddings, TradeS4OneHot
from src.dataloaders.trade_grid import TradeS4ND2D, TradeS4ND2DState, TradeS4ND3D, TradeS4ND4D

root = Path.cwd().parents[1]
spec = importlib.util.spec_from_file_location('target_selection', root / 'src/utils/s4_benchmark.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)
report = []
# A root containing only the aggregated release verifies there is no fallback
# to raw imports/exports or to the checkout when a source root is supplied.
with tempfile.TemporaryDirectory() as folder:
    data_root = Path(folder)
    destination = data_root / 'data/processed'
    destination.mkdir(parents=True)
    (destination / 'hs2_monthly_aggregated.csv').symlink_to(root / 'data/processed/hs2_monthly_aggregated.csv')
    for cls in [TradeS4Embeddings, TradeS4OneHot, TradeS4ND2D, TradeS4ND2DState, TradeS4ND3D, TradeS4ND4D]:
        ds = cls(cls._name_, data_dir=str(data_root), input_len=36)
        ds.setup()
        assert Path(ds.pipe.config.aggregated_csv).parent == data_root.resolve() / 'data/processed'
        # A deliberately wrong nominal size must not change the month policy.
        loader = ds.train_dataloader(batch_size=7, num_workers=0, drop_last=False)
        batches = list(loader.batch_sampler)
        assert len(loader) == 132
        assert sorted(i for b in batches for i in b) == list(range(len(loader.dataset)))
        batch = next(iter(loader))
        x, y, extra = batch
        _, valid = selection.select_valid_targets(y, y, ds, extra)
        assert valid.numel() // 2 == 918, (cls._name_, valid.shape)
        sizes = sorted(set(map(len, batches)))
        assert len(sizes) == 1
        report.append({'model': cls._name_, 'batch_size': sizes[0], 'steps': len(loader),
                       'valid_targets_per_step': 918, 'samples': len(loader.dataset)})
        print(json.dumps(report[-1]), flush=True)
        del ds, loader, batches, batch, x, y, extra, valid
        gc.collect()
print('ALL SIX S4 DATA PATHS MATCH', flush=True)
