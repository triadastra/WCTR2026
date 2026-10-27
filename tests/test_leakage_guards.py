"""Exercise preprocessing guards without requiring the optional Torch runtime.

Load the actual preprocessing definitions via AST; no model or tensor methods
are called. This lets these checks run on lightweight data-audit machines.
"""
import ast
import __future__
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import typing
import math
import re

import numpy as np
import pandas as pd
import unittest

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    ns = dict(vars(typing), np=np, pd=pd, dataclass=dataclass, Path=Path, math=math, re=re,
              KEYS=["State", "Commodity", "Import/Export"],
              VALUE_CANON="Value", WEIGHT_CANON="Weight")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), ns)
    return ns


def pipeline(**overrides):
    ns = definitions("src/data/dataloader.py", ["DataConfig", "TradeDataPipeline"])
    cfg = ns["DataConfig"](**overrides)
    pipe = ns["TradeDataPipeline"](cfg)
    times = pd.date_range("2023-01-01", "2025-05-01", freq="MS")
    pipe.df_full = pd.DataFrame({"Time": times, "State": "A", "Commodity": "C",
                                "Import/Export": "Import", "Country": "X",
                                "Value": np.arange(len(times)) + 1.,
                                "Weight": np.arange(len(times)) + 2.})
    return pipe


def check_main_rejects_invalid_splits(overrides):
    with unittest.TestCase().assertRaisesRegex(ValueError, "Split dates must satisfy|must be before|must be on or before|valid month|overlap|month-start|monthly date"):
        pipeline(**overrides).create_splits()


def check_s4_rejects_overlap_and_accepts_valid_splits(path):
    split = definitions(path, ["temporal_split_calendar"])["temporal_split_calendar"]
    cfg = SimpleNamespace(split_train_end="2023-12-01", split_val_start="2024-01-01",
                          split_val_end="2024-12-01", split_test_start="2025-01-01",
                          split_test_end="2025-05-01")
    grid = pipeline().df_full
    tr, va, te = split(grid, cfg)
    assert tr.Time.max() < va.Time.min() <= va.Time.max() < te.Time.min()
    cfg.split_train_end = cfg.split_val_start
    with unittest.TestCase().assertRaisesRegex(ValueError, "Split dates must satisfy"):
        split(grid, cfg)


def check_legacy_cohort_is_rejected():
    pipe = pipeline(cohort_from_train_only=False)
    with unittest.TestCase().assertRaisesRegex(ValueError, "cohort_from_train_only must be True"):
        pipe.create_splits()
    with unittest.TestCase().assertRaisesRegex(ValueError, "cohort_from_train_only must be True"):
        pipe.filter_sparse_combos(pipe.df_full)


def check_held_out_values_cannot_change_training():
    a = pipeline(start="2023-01-01", lag_count=1, filter_sparse=False)
    b = pipeline(start="2023-01-01", lag_count=1, filter_sparse=False)
    held_out = b.df_full.Time > pd.Timestamp(b.config.train_end)
    b.df_full.loc[held_out, ["Value", "Weight"]] *= 100000
    a.create_splits()
    b.create_splits()
    pd.testing.assert_frame_equal(a.df_train, b.df_train)
    pd.testing.assert_frame_equal(a.stats["exact"], b.stats["exact"])


class LeakageGuards(unittest.TestCase):
    def test_invalid_splits(self):
        for overrides in [
            {"train_end": "2024-01-01"}, {"val_end": "2025-01-01"},
            {"val_start": "2024-12-01", "val_end": "2024-01-01"},
            {"test_end": "2024-12-01"}, {"val_start": "NaT"},
        ]:
            with self.subTest(overrides=overrides):
                check_main_rejects_invalid_splits(overrides)

    def test_s4_splits(self):
        for path in ["external/s4/shared_dataloader.py",
                     "external/s4/src/dataloaders/shared_dataloader.py"]:
            with self.subTest(path=path):
                check_s4_rejects_overlap_and_accepts_valid_splits(path)

    def test_legacy_cohort(self):
        check_legacy_cohort_is_rejected()

    def test_held_out_invariance(self):
        check_held_out_values_cannot_change_training()


if __name__ == "__main__":
    unittest.main()
