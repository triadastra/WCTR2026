"""Publication completeness and data-derived sensitivity reporting."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_paper_results as builder
from src.utils.exp3_summary import summarize_sweep, sweep_summary_text
from src.utils.experiment_manifest import load_manifest


@pytest.mark.parametrize("missing", [None, "model", "seed"])
def test_sweep_collector_requires_manifest_cells_and_seeds(tmp_path, monkeypatch, missing):
    from src.utils import exp3_protocol, training_completion

    expected = exp3_protocol.coordinates()
    monkeypatch.setattr(builder, "corrected_per_combo_artifact", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(builder, "corrected_prediction_artifact", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(exp3_protocol, "validate_contract", lambda *a: (True, "ok"))
    monkeypatch.setattr(training_completion, "validate_completion_artifact", lambda *a: None)
    for model, window, lag, seed in expected:
        if missing == "model" and model == "s4_bidir":
            continue
        if missing == "seed" and (model, window, lag, seed) == expected[-1]:
            continue
        directory = tmp_path / exp3_protocol.run_name(model, window, lag, seed)
        directory.mkdir()
        (directory / "predictions.csv").write_text("fixture for roster-only test\n")
        (directory / "metrics.json").write_text(json.dumps({
            "model": model, "variant": "embeddings", "seed": seed,
            "predictions_file": "predictions.csv",
            "data_protocol": {"input_len": window, "lag_count": lag,
                              "train_target_start": "2015-01-01"},
            "metrics": {"norm_mse_combined": 0.1},
        }))
    if missing:
        with pytest.raises(RuntimeError, match="Exp 3 sweep incomplete"):
            builder._sweep_rows(tmp_path)
    else:
        rows = builder._sweep_rows(tmp_path)
        assert len(rows) == len(expected) == 192
        assert {(r["model"], r["window"], r["lag_count"], r["seed"]) for r in rows} == set(expected)


@pytest.fixture
def small_grid():
    manifest = {"PAPER_SWEEP_MODELS": "lstm", "PAPER_WINDOWS": "6 12",
                "PAPER_LAGS": "6 12", "PAPER_SEEDS": "26 105"}
    rows = [dict(model="lstm", variant="embeddings", window=window, lag_count=lag,
                 seed=seed, norm_mse_combined=mean + offset)
            for (window, lag), mean in [((6, 6), 10), ((6, 12), 20),
                                       ((12, 6), 30), ((12, 12), 40)]
            for seed, offset in [(26, -1), (105, 1)]]
    return rows, manifest


def test_summary_matches_hand_calculated_cell_and_marginal_values(small_grid):
    rows, manifest = small_grid
    summary, marginals = summarize_sweep(rows, manifest)
    row = summary[0]
    assert (row["cell_mean_min"], row["cell_mean_max"], row["cell_mean_range"]) == (10, 40, 30)
    assert row["cell_mean_range_pct_of_min"] == 300
    assert row["lowest_observed_test_cells"] == "w6_l6"
    assert row["cell_seed_sample_sd_min"] == pytest.approx(math.sqrt(2))
    assert row["window_marginal_range"] == 20
    assert row["lag_count_marginal_range"] == 10
    assert [r["mean_norm_mse_combined"] for r in marginals] == [15, 35, 20, 30]


def test_observations_change_when_results_reverse(small_grid):
    rows, manifest = small_grid
    before, _ = summarize_sweep(rows, manifest)
    for row in rows:
        row["norm_mse_combined"] = 100 - row["norm_mse_combined"]
    after, _ = summarize_sweep(rows, manifest)
    assert after[0]["lowest_observed_test_cells"] == "w12_l12"
    assert sweep_summary_text(before) != sweep_summary_text(after)
    assert "w12_l12" in sweep_summary_text(after)
    assert "no systematic effect" not in sweep_summary_text(after)


def test_summary_keeps_ties_and_zero_denominators_honest(small_grid):
    rows, manifest = small_grid
    for row in rows:
        row["norm_mse_combined"] = 0
    summary, _ = summarize_sweep(rows, manifest)
    assert summary[0]["lowest_observed_test_cells"] == "w6_l6; w6_l12; w12_l6; w12_l12"
    assert summary[0]["cell_mean_range_pct_of_min"] is None
    assert "undefined because the minimum is zero" in sweep_summary_text(summary)


def test_summary_does_not_select_on_rounded_losses(small_grid):
    rows, manifest = small_grid
    for row in rows:
        row["norm_mse_combined"] = 1.0
    for row in rows:
        if (row["window"], row["lag_count"]) == (12, 12):
            row["norm_mse_combined"] -= 1e-10
    summary, _ = summarize_sweep(rows, manifest)
    assert summary[0]["lowest_observed_test_cells"] == "w12_l12"
    assert summary[0]["cell_mean_range"] > 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_seed", "wrong_variant", "nan", "negative"])
def test_summary_refuses_incomplete_or_invalid_evidence(small_grid, mutation):
    rows, manifest = deepcopy(small_grid)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "wrong_seed":
        rows[0]["seed"] = 999
    elif mutation == "wrong_variant":
        rows[0]["variant"] = "onehot"
    else:
        rows[0]["norm_mse_combined"] = float("nan") if mutation == "nan" else -1
    with pytest.raises(ValueError):
        summarize_sweep(rows, manifest)


@pytest.mark.parametrize("create_directory", [False, True])
def test_absent_or_empty_sweep_cannot_publish(tmp_path, monkeypatch, create_directory):
    if create_directory:
        (tmp_path / "outputs/sweep").mkdir(parents=True)
    output = tmp_path / "results.xlsx"
    output.write_bytes(b"previous workbook")
    monkeypatch.setattr(builder, "_exp1_rows", lambda _: ([], {}))
    monkeypatch.setattr(builder, "_exp2_rows", lambda _: [])
    monkeypatch.setattr(builder.sys, "argv", ["build_paper_results.py", "--repo", str(tmp_path)])
    with pytest.raises(RuntimeError, match="Exp 3 sweep incomplete"):
        builder.main()
    assert output.read_bytes() == b"previous workbook"


def test_workbook_observations_are_computed_from_its_sweep(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    manifest = load_manifest(repo / "config/experiment_manifest.env")
    rows = [dict(model=model, variant="embeddings", window=int(w), lag_count=int(lag),
                 seed=int(seed), mode="one_step_normalized",
                 norm_mse_combined=(i + 1) * .01 + int(w) * .001 + int(lag) * .0001 + j * .00001)
            for i, model in enumerate(manifest["PAPER_SWEEP_MODELS"].split())
            for w in manifest["PAPER_WINDOWS"].split() for lag in manifest["PAPER_LAGS"].split()
            for j, seed in enumerate(manifest["PAPER_SEEDS"].split())]
    stub = [dict(model="random_walk", variant="none", seed=26, mode="one_step_normalized",
                 norm_mse_combined=.1234567890123)]
    from src.utils.report_metadata import attach_report_evidence
    payload = {**stub[0], "predictions_file": "predictions.csv",
               "metrics": {"norm_mse_combined": .1234567890123},
               "run_contract": {"model_config": {"description": "=1+1", "missing": None}}}
    (tmp_path / "predictions.csv").write_text("metadata-only fixture\n")
    (tmp_path / "metrics.json").write_text(json.dumps(payload))
    attach_report_evidence(stub[0], payload, tmp_path / "metrics.json")
    monkeypatch.setattr(builder, "_exp1_rows", lambda _: (stub, manifest))
    monkeypatch.setattr(builder, "_exp2_rows", lambda _: stub)
    monkeypatch.setattr(builder, "_sweep_rows", lambda _: rows)
    monkeypatch.setattr(builder, "_ensemble", lambda repo, full_dir: pd.DataFrame())
    monkeypatch.setattr(builder.sys, "argv", ["build_paper_results.py", "--repo", str(tmp_path)])
    builder.main()
    sheets = pd.read_excel(tmp_path / "results.xlsx", sheet_name=None)
    observations = sheets["README"].set_index("key").loc["Exp3 observations", "value"]
    expected, _ = summarize_sweep(rows, manifest)
    assert observations == sweep_summary_text(expected)
    assert len(sheets["Exp3_Sweep"]) == 192
    assert len(sheets["Exp3_Summary"]) == 4
    assert len(sheets["Exp3_Marginals"]) == 32
    assert "not uniform container demand" in sheets["README"].set_index("key").loc["Target definition", "value"]
    assert sheets["Exp1_Aggregated"].iloc[0].norm_mse_combined == pytest.approx(.1234567890123, abs=1e-15)
    assert len(sheets["Run_Details"]) == 2
    assert "=1+1" in set(sheets["Run_Values"].value)
    assert "Evaluation_Scope" in sheets and "Metric_Definitions" in sheets
