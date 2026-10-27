"""The report scripts must fail closed: complete seed rosters, identical
populations, one declared arm per figure cell, and no cached-CSV bypass.
Ported from Astra 6's 2026-09-09 fix set into the origin/main line."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "util"))

import make_significance as sig  # noqa: E402
import make_vs_baseline as baseline  # noqa: E402


def _panel(states=("A", "B"), commodities=("c1", "c2"), months=3, offset=0.0):
    keys = [(s, c, "Import", f"2025-{m:02d}-01")
            for s in states for c in commodities for m in range(1, months + 1)]
    index = pd.MultiIndex.from_tuples(keys, names=sig.KEY)
    return pd.Series(np.arange(len(index)) / 10.0 + offset, index=index)


def test_average_seed_losses_refuses_an_incomplete_roster():
    losses = {s: _panel() for s in sig.SEEDS[:2]}
    with pytest.raises(RuntimeError, match="incomplete paired seed roster"):
        sig.average_seed_losses("lstm_embeddings", losses)


def test_average_seed_losses_refuses_a_differing_population():
    losses = {s: _panel() for s in sig.SEEDS}
    losses[sig.SEEDS[-1]] = losses[sig.SEEDS[-1]].iloc[:-1]
    with pytest.raises(RuntimeError, match="population differs"):
        sig.average_seed_losses("lstm_embeddings", losses)


def test_average_seed_losses_averages_a_complete_roster():
    losses = {s: _panel(offset=i) for i, s in enumerate(sig.SEEDS)}
    avg, means, index = sig.average_seed_losses("lstm_embeddings", losses)
    assert index.equals(losses[sig.SEEDS[0]].sort_index().index)
    assert means == pytest.approx([float(v.mean()) for v in losses.values()])
    assert avg.iloc[0] == pytest.approx(1.0)  # (0 + 1 + 2) / 3


def test_aligned_difference_refuses_to_intersect():
    left, right = _panel(), _panel().iloc[:-1]
    with pytest.raises(RuntimeError, match="Paired populations differ"):
        sig.aligned_difference(left, right)
    np.testing.assert_allclose(sig.aligned_difference(_panel(offset=1), _panel()), 1.0)


def test_s4nd_layouts_get_separate_cells_and_collisions_raise():
    rows = [dict(model="s4nd", variant=v, improvement_pct=i, mean_metric=1.0,
                 sig_raw=False, sig_fdr=False) for i, v in enumerate(["2d", "2d_state"])]
    _, _, matrix, *_ = baseline._grid_from_rows(rows)
    assert np.isfinite(matrix).sum() == 2
    assert baseline.STRUCT["2d"] != baseline.STRUCT["2d_state"]
    with pytest.raises(RuntimeError, match="Multiple arms map to one heatmap cell"):
        baseline._grid_from_rows(rows + rows[:1])


def test_plot_only_bypass_is_gone(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, "OUTDIR", tmp_path)
    monkeypatch.setattr(baseline.sys, "argv", ["make_vs_baseline.py", "--plot-only"])
    monkeypatch.setattr(baseline, "load_model_losses",
                        lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    assert baseline.main() == 2


def test_significance_no_longer_writes_into_the_workbook():
    assert not hasattr(sig, "embed_significance_sheet")
    src = (ROOT / "util" / "make_significance.py").read_text()
    assert "load_workbook" not in src


# --- the declared 52-arm family (src/utils/exp2_roster.py) ---

def test_roster_is_the_57_arm_family_that_run_full_test_launches():
    from src.utils import exp2_roster as roster
    assert len(roster.EXP2_ARMS) == 57
    from src.utils.exp2_protocol import expected_runs as protocol_runs
    from src.utils.experiment_manifest import load_manifest
    runs = protocol_runs(load_manifest(ROOT / "config/experiment_manifest.env"))
    assert set(roster.EXP2_ARMS) == {(run.model, run.variant) for run in runs}
    assert set(roster.expected_runs()) == {run.directory_name for run in runs}
    assert len(roster.expected_runs()) == 171
    assert "s4_s4nd_2d_state_seed26" in roster.expected_runs()
    assert "s4_cross_attention_2d_seed26" in roster.expected_runs()
    assert roster.expected_runs()["s4_s4_embeddings_seed105"] == ("s4", "embeddings", 105)


def test_check_roster_refuses_partial_or_stray_directories():
    from src.utils.exp2_roster import check_roster, expected_runs
    full = set(expected_runs())
    assert check_roster(full) == expected_runs()
    with pytest.raises(RuntimeError, match="missing 1"):
        check_roster(full - {"lstm_embeddings_seed26"})
    with pytest.raises(RuntimeError, match="unexpected 1"):
        check_roster(full | {"s4_smoke_seed42"})


def test_loader_refuses_an_inexact_roster_before_reading_payloads(tmp_path, monkeypatch):
    for name in ("lstm_embeddings_seed26", "lstm_embeddings_seed105"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text("{}")
    monkeypatch.setattr(sig, "RUNS", tmp_path)
    with pytest.raises(RuntimeError, match="Inexact Exp 2 roster"):
        sig.load_model_losses()


def test_loader_refuses_a_payload_whose_identity_disagrees_with_its_directory(tmp_path, monkeypatch):
    import json
    from src.utils.exp2_roster import expected_runs
    for name, (m, v, s) in expected_runs().items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(json.dumps({"model": m, "variant": v, "seed": s}))
    # copy a random-walk payload into an LSTM directory
    (tmp_path / "lstm_embeddings_seed105" / "metrics.json").write_text(
        json.dumps({"model": "random_walk", "variant": "none", "seed": 26}))
    monkeypatch.setattr(sig, "RUNS", tmp_path)
    with pytest.raises(RuntimeError, match="does not match the roster entry"):
        sig.load_model_losses()
