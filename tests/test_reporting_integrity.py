"""Adversarial numerical and publication checks, including failure preservation."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import build_paper_results as builder
from src.inference.metrics import REQUIRED_EVALUATION_METRICS, SCALED_METRIC_NAMES
from src.utils import exp2_artifacts, scaled_metric_validation
from src.utils.model_measurements import model_measurements
from src.utils.report_metadata import attach_report_evidence, reporting_sheets, verify_report_snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_seed_statistics_retain_sub_display_precision_and_counts_are_not_averaged():
    rows = [dict(model="lstm", variant="embeddings", seed=seed,
                 norm_mse_combined=value, num_params=7)
            for seed, value in [(26, 1e-10), (105, 2e-10), (115, 3e-10)]]
    summary = builder._aggregate_seeds(rows).iloc[0]
    assert summary.norm_mse_combined == pytest.approx(2e-10, abs=1e-20)
    assert summary.norm_mse_std == pytest.approx(1e-10, abs=1e-20)
    assert summary.num_params == 7
    rows[-1]["num_params"] = 8
    with pytest.raises(RuntimeError, match="num_params"):
        builder._aggregate_seeds(rows)


@pytest.fixture
def ensemble(tmp_path, monkeypatch):
    artifacts = []
    for seed, error in [(26, -2.), (105, 1.), (115, 4.)]:
        directory = tmp_path / "outputs/full" / f"lstm_embeddings_seed{seed}"
        directory.mkdir(parents=True)
        path = directory / "metrics.json"
        payload = {"predictions_file": "predictions.csv"}
        path.write_text(json.dumps(payload))
        truth = np.array([1., 2., 3., 4., 5.])
        pd.DataFrame({"State": "A", "Commodity": "01", "Import/Export": "Import",
                      "Time": pd.date_range("2025-01-01", periods=5, freq="MS").strftime("%Y-%m-%d"),
                      "true_value": truth, "true_weight": truth * 2,
                      "pred_value": truth + error, "pred_weight": truth * 2 + error,
                      "true_value_real": truth * 10, "true_weight_real": truth * 20,
                      "pred_value_real": (truth + error) * 10,
                      "pred_weight_real": (truth * 2 + error) * 10}).sample(frac=1, random_state=seed).to_csv(
                          directory / "predictions.csv", index=False)
        artifacts.append((SimpleNamespace(model="lstm", variant="embeddings", seed=seed), path, payload))
    monkeypatch.setattr(exp2_artifacts, "validated_exp2_artifacts", lambda *a, **k: artifacts)
    def scales(payload, predictions, targets, keys, **kwargs):
        assert {k[1] for k in keys} == {"01"}
        assert len(kwargs["identities"]) == 5
        return {name: float(np.mean(np.abs(predictions - targets))) for name in SCALED_METRIC_NAMES}
    monkeypatch.setattr(scaled_metric_validation, "source_scaled_metrics", scales)
    return tmp_path, tmp_path / "outputs/full", artifacts


def test_ensemble_scores_mean_predictions_and_complete_metrics(ensemble):
    repo, directory, _ = ensemble
    row = builder._ensemble(repo, directory).iloc[0]
    assert row.norm_mse_combined == 1.0  # ((-2 + 1 + 4)/3)^2, not mean seed MSE 7
    assert row.mase == 1.0
    assert set(REQUIRED_EVALUATION_METRICS).issubset(row.index)
    assert (row.n_members, row.n_targets) == (3, 5)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_truth", "wrong_key"])
def test_ensemble_never_silently_drops_bad_members(ensemble, mutation):
    repo, directory, artifacts = ensemble
    path = artifacts[-1][1].parent / "predictions.csv"
    if mutation == "missing":
        path.unlink()
    else:
        frame = pd.read_csv(path, dtype={"Commodity": str})
        if mutation == "duplicate":
            frame = pd.concat([frame, frame.iloc[:1]])
        elif mutation == "wrong_truth":
            frame.loc[0, "true_value"] += 1
        else:
            frame.loc[0, "State"] = "B"
        frame.to_csv(path, index=False)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        builder._ensemble(repo, directory)


def test_metadata_preserves_nested_settings_and_refuses_changed_files(tmp_path):
    payload = dict(model="lstm", variant="embeddings", seed=26, predictions_file="predictions.csv",
                   metrics={"num_params": 17, "norm_mse_combined": .1234567890123},
                   run_contract={"model_config": {"hidden_size": 128, "custom": [None, False, "=1+1"]}},
                   architecture={"bidirectional": False})
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload))
    (tmp_path / "predictions.csv").write_text("original bytes")
    row = attach_report_evidence({k: payload[k] for k in ("model", "variant", "seed")}, payload, path)
    sheets = reporting_sheets({"Exp2": [row]})
    fields = sheets["Run_Values"].set_index("field")
    assert fields.loc["run_contract.model_config.hidden_size", "value"] == 128
    assert fields.loc["run_contract.model_config.custom[0]", "value"] == "not recorded"
    assert fields.loc["metrics.norm_mse_combined", "value"] == .1234567890123
    verify_report_snapshot([row])
    (tmp_path / "predictions.csv").write_text("new bytes")
    with pytest.raises(RuntimeError, match="changed during build"):
        verify_report_snapshot([row])


def test_parameter_measurements_count_frozen_shared_tensors_and_buffers():
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    model.register_buffer("scale", torch.ones(3))
    measured = model_measurements(model, "lstm")
    assert measured["total_parameters"] == 8
    assert measured["trainable_parameters"] == 6
    assert measured["frozen_parameters"] == 2
    assert measured["parameter_bytes"] == 32 and measured["buffer_bytes"] == 12
    json.dumps(measured, allow_nan=False)


@pytest.mark.parametrize("count", [float("nan"), -1, 2.5, True, "unknown"])
def test_invalid_parameter_counts_are_not_silently_blanked(tmp_path, count):
    payload = {"metrics": {"num_params": count}}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        attach_report_evidence({}, payload, path)


@pytest.mark.parametrize("family", ["xgboost", "random_forest", "random_walk"])
def test_non_neural_summaries_do_not_claim_inherited_adam_or_hidden_layers(tmp_path, family):
    payload = dict(model=family, variant="embeddings", seed=26, predictions_file="predictions.csv",
                   run_contract={"model_config": {"hidden_size": 128, "num_layers": 4,
                                                  "learning_rate": .05, "n_estimators": 100},
                                 "training_config": {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 1e-4}})
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload))
    (tmp_path / "predictions.csv").write_text("fixture")
    row = attach_report_evidence({k: payload[k] for k in ("model", "variant", "seed")}, payload, path)
    detail = reporting_sheets({"Exp2": [row]})["Run_Details"].iloc[0]
    assert detail.optimizer != "adamw"
    assert detail.hidden_size is None and detail.layers is None and detail.weight_decay is None
    if family == "xgboost":
        assert detail.initial_lr == .05
    else:
        assert detail.initial_lr is None


def test_tree_measurements_are_json_safe_and_not_zero_parameters():
    from sklearn.ensemble import RandomForestRegressor
    tree = RandomForestRegressor(n_estimators=2, random_state=26).fit([[0], [1], [2]], [0, 1, 2])
    measured = model_measurements(SimpleNamespace(model_value=tree, model_weight=tree), "random_forest")
    assert measured["value"]["trees"] == 2
    assert measured["value"]["nodes"] >= measured["value"]["leaves"] > 0
    assert "total_parameters" not in measured
    json.dumps(measured, allow_nan=False)


def test_graphs_reject_cached_csv_when_live_validation_fails(tmp_path, monkeypatch):
    from scripts import build_graphs
    (tmp_path / "results.csv").write_text("model,variant,norm_mse_combined\nlstm,embeddings,0\n")
    monkeypatch.setattr(build_graphs, "RUNS", tmp_path)
    def reject(*args):
        raise RuntimeError("source mismatch")
    monkeypatch.setattr(build_graphs, "_exp2_rows", reject)
    with pytest.raises(RuntimeError, match="source mismatch"):
        build_graphs.load_results()


@pytest.mark.parametrize("directory,model,variant", [
    ("seasonal_naive_none_seed26", "random_walk", "none"),
    ("lstm_embeddings_seed26", "lstm", "onehot"),
])
def test_csv_refuses_relabelled_model_or_variant_without_replacing_output(tmp_path, monkeypatch, directory, model, variant):
    from scripts import collect_results
    run = tmp_path / directory
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps(dict(model=model, variant=variant, seed=26)))
    output = tmp_path / "results.csv"
    output.write_text("previous results\n")
    monkeypatch.setattr(sys, "argv", ["collect_results.py", "--runs-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="model or variant mismatch"):
        collect_results.main()
    assert output.read_text() == "previous results\n"


def test_all_external_s4_adapters_exclude_future_targets():
    result = subprocess.run([sys.executable, str(ROOT / "tests/check_s4_leakage.py")],
                            cwd=ROOT / "external/s4", capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL S4 LEAKAGE CHECKS PASSED" in result.stdout


@pytest.mark.parametrize("claim", ["real-time", "confirmatory"])
def test_unsubstantiated_leakage_free_claim_cannot_publish(tmp_path, monkeypatch, claim):
    output = tmp_path / "results.xlsx"
    output.write_bytes(b"previous workbook")
    monkeypatch.setattr(sys, "argv", ["build_paper_results.py", "--repo", str(tmp_path), "--claim", claim])
    with pytest.raises(RuntimeError, match="no audited"):
        builder.main()
    assert output.read_bytes() == b"previous workbook"


@pytest.mark.parametrize("variant", ["embeddings", "onehot"])
def test_full_window_tree_inputs_exclude_each_test_target_and_later_values(variant):
    from src.data.dataloader import DataConfig, TradeDataPipeline
    from src.models.tabular_common import extract_tabular_xy
    months = pd.date_range("2017-01-01", "2021-05-01", freq="MS")
    source = pd.DataFrame([dict(State=state, Commodity="01", Time=month,
                               Value=(i+1)*scale, Weight=(i+2)*scale,
                               **{"Import/Export": "Import"})
                           for state, scale in [("A", 1), ("B", 2)]
                           for i, month in enumerate(months)])
    def build(frame):
        pipe = TradeDataPipeline(DataConfig(
            start="2017-01-01", end="2021-05-01", input_len=3, lag_count=2,
            train_end="2019-12-01", val_start="2020-01-01", val_end="2020-12-01",
            test_start="2021-01-01", test_end="2021-05-01", tabular_feature_mode="window"))
        pipe.df_full = frame
        pipe.create_splits()
        loader = pipe.get_dataloaders(batch_size=2, num_workers=0, shuffle_train=False)[2]
        x, y, names = extract_tabular_xy(
            loader, variant=variant, num_numeric_features=len(pipe.feat_cols),
            num_states=2, num_commodities=1, num_flows=1)
        return pipe, loader, x, y, names
    original, loader, x, y, names = build(source)
    times = pd.DatetimeIndex([s["target_time"] for s in loader.dataset.samples])
    for cutoff in pd.date_range("2021-01-01", "2021-05-01", freq="MS"):
        edited = source.copy()
        edited.loc[edited.Time >= cutoff, ["Value", "Weight"]] *= 10000
        pipe, _, changed_x, changed_y, changed_names = build(edited)
        assert original.kept_combos == pipe.kept_combos
        pd.testing.assert_frame_equal(original.stats["exact"], pipe.stats["exact"], check_exact=True)
        assert names == changed_names
        np.testing.assert_array_equal(x[times <= cutoff], changed_x[times <= cutoff])
        assert not np.array_equal(y[times == cutoff], changed_y[times == cutoff])
