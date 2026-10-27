"""explain_combo: completeness on CPU, and the MPS path (no float64 there)."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import explain_exp2  # noqa: E402

L, G, F = 6, 4, 5
FEATS = ["value", "weight", "value_lag_1", "sin_month", "cos_month"]


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(L * G * F, G * 2)

    def forward(self, x, *_, group_mask=None):
        return torch.tanh(self.w(x.flatten(1))).view(-1, G, 2)


def _explain(device):
    torch.manual_seed(0)

    def item(t):
        return {"x_numeric": torch.randn(L, G, F), "group_mask": torch.ones(G), "target_time": t}
    train = [item(pd.Timestamp("2020-01-01") + pd.DateOffset(months=i)) for i in range(10)]
    test = [item(pd.Timestamp("2025-01-01") + pd.DateOffset(months=i)) for i in range(2)]
    loaders = [types.SimpleNamespace(dataset=train), None, types.SimpleNamespace(dataset=test)]
    pipe = types.SimpleNamespace(combo2id={("A", str(c), "X"): c for c in range(G)}, feat_cols=FEATS)
    lanes = [("A", "0", "X"), ("A", "2", "X")]
    times = [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")]
    device = torch.device(device)
    return explain_exp2.explain_combo(_Net().to(device), "transformer_linhead", pipe, loaders,
                                      lanes, times, device, bg_n=4, steps=8)


def test_combo_attributions_are_complete_on_cpu():
    r = _explain("cpu")
    total = r["own"].sum(axis=(3, 4)) + r["src"].sum(axis=(3, 4))
    assert np.abs(total - (r["fx"] - r["base"])).max() < 1e-6


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_combo_runs_on_mps_and_matches_cpu():
    cpu, mps = _explain("cpu"), _explain("mps")
    for key in ("own", "src", "fx", "base"):
        np.testing.assert_allclose(mps[key], cpu[key], atol=1e-6)
