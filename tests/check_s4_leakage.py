"""Run inside external/s4: perturb future outcomes through every live adapter."""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd()))
import src.dataloaders.trade_unified as unified
import src.dataloaders.trade_grid as grid
from s4_test_support import synthetic


def run():
    torch.set_num_threads(1)
    cutoff = pd.Timestamp("2025-03-01")
    changed = False

    def source(root, length, lags, aggregate=False):
        pipe = synthetic(root, length, lags)
        pipe.config.aggregate = aggregate
        pipe.config.batch_by_target_month = True
        if changed:
            pipe.df_full.loc[pipe.df_full.Time >= cutoff, ["Value", "Weight"]] *= 10000
        pipe.create_splits()
        return pipe

    unified._canonical_pipeline = source
    grid._clean_pipeline = lambda root, length: source(root, length, 2)
    classes = [unified.TradeS4Embeddings, unified.TradeS4OneHot,
               grid.TradeS4ND2DState, grid.TradeS4ND2D, grid.TradeS4ND3D, grid.TradeS4ND4D]
    for cls in classes:
        before = cls(cls._name_, data_dir=".", input_len=3)
        before.setup()
        changed = True
        after = cls(cls._name_, data_dir=".", input_len=3)
        after.setup()
        for method in ("train_dataloader", "val_dataloader"):
            a = getattr(before, method)(batch_size=1, num_workers=0, shuffle=False)
            b = getattr(after, method)(batch_size=1, num_workers=0, shuffle=False)
            assert len(a) == len(b)
            for original, edited in zip(a, b):
                for x, y in zip(original[:2], edited[:2]):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
        a = list(before.test_dataloader(batch_size=1, num_workers=0, shuffle=False))
        b = list(after.test_dataloader(batch_size=1, num_workers=0, shuffle=False))
        assert len(a) == len(b)
        # Every group has five test months in flat adapters. ND evaluation
        # samples are month-major with a layout-dependent number of subgrids.
        flat = issubclass(cls, unified._TradePerSeriesBase)
        for i, (original, edited) in enumerate(zip(a, b)):
            month_index = i % 5 if flat else i // (len(a) // 5)
            if month_index <= 2:
                torch.testing.assert_close(original[0], edited[0], rtol=0, atol=0)
            if month_index == 2:
                assert not torch.equal(original[1], edited[1]), "mutation must reach the target"
        print(json.dumps({"adapter": cls._name_, "future_isolation": "passed"}))
        changed = False
    print("ALL S4 LEAKAGE CHECKS PASSED")


if __name__ == "__main__":
    run()
