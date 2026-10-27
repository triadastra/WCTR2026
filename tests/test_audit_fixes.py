"""Regression tests for the 2026-09-03 audit fixes.

Each test pins one defect that silently changed a published number or a
published label. They are grouped here rather than scattered so the fix pass
has a single place to fail from.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import create_model  # noqa: E402
from src.models.attention import CrossAttentionModule  # noqa: E402


# ---------------------------------------------------------------------------
# Tree models must take their RNG seed from the run, not a hardcoded 42.
#
# All three hardcoded random_state=42, so the 3-seed protocol produced three
# byte-identical models and every tree arm reported a seed std of exactly 0.
# ---------------------------------------------------------------------------

TREE_ARMS = [
    ("xgboost", "xgb_params"),
    ("lightgbm", "lgbm_params"),
    ("random_forest", "rf_params"),
]


def _make_tree(name, **kwargs):
    return create_model(
        name=name,
        variant="embeddings",
        num_numeric_features=28,
        num_states=8,
        num_commodities=93,
        num_flows=2,
        **kwargs,
    )


@pytest.mark.parametrize("name,attr", TREE_ARMS)
def test_tree_random_state_is_configurable(name, attr):
    for seed in (26, 105, 115):
        params = getattr(_make_tree(name, random_state=seed), attr)
        assert params["random_state"] == seed, (
            f"{name} ignored random_state={seed}; a hardcoded value collapses "
            f"the 3-seed protocol to a single run"
        )


@pytest.mark.parametrize("name,attr", TREE_ARMS)
def test_tree_random_state_defaults_to_42_when_unset(name, attr):
    """The default is unchanged, so an explicit config still reproduces old runs."""
    assert getattr(_make_tree(name), attr)["random_state"] == 42


@pytest.mark.parametrize("script", ["train", "inference"])
def test_launchers_pass_the_run_seed_to_tree_models(script):
    """The constructor arg is useless unless the launchers actually set it."""
    src = (ROOT / "scripts" / f"{script}.py").read_text()
    assert 'model_cfg.setdefault("random_state", args.seed)' in src, (
        f"scripts/{script}.py must seed the tree arms from --seed; without it "
        f"the run seed never reaches the booster's own RNG"
    )
    # setdefault, not assignment: an explicit config value must still win.
    assert 'model_cfg["random_state"] = args.seed' not in src


# ---------------------------------------------------------------------------
# Single-key "cross attention" is a fixed linear map, and must say so.
# ---------------------------------------------------------------------------

def test_single_key_cross_attention_warns():
    m = CrossAttentionModule(d_query=42, d_key=42, d_value=42, num_heads=4, d_k=8)
    x = torch.randn(3, 1, 42)  # ONE key token
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m(x, x, x)
    assert any("single key token" in str(w.message) for w in caught), (
        "a one-key attention module must announce that W_q/W_k are untrainable"
    )


def test_single_key_cross_attention_leaves_wq_wk_untrained():
    """Not an initialization artifact: the gradient is zero at EVERY step."""
    m = CrossAttentionModule(d_query=42, d_key=42, d_value=42, num_heads=4, d_k=8)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(3):
        x = torch.randn(4, 1, 42)
        opt.zero_grad()
        m(x, x, x).sum().backward()
        assert float(m.W_q.weight.grad.abs().max()) == 0.0
        assert float(m.W_k.weight.grad.abs().max()) == 0.0
        # W_v / W_o are the map that actually trains.
        assert float(m.W_v.weight.grad.abs().max()) > 0.0
        opt.step()


@pytest.mark.parametrize("variant", ["cross_attention", "film_attention"])
def test_transformer_flat_attention_variants_train_query_and_key(variant):
    """Corrected per-series attention must train query/key projections."""
    model = create_model(
        name="transformer", variant=variant, num_numeric_features=28,
        num_states=8, num_commodities=93, num_flows=2,
        hidden_size=32, num_layers=1, num_heads=4, d_ff=64,
    )
    assert getattr(model, "requires_combo_loader", False) is False
    out = model(
        torch.randn(4, 12, 28),
        torch.randint(0, 8, (4,)),
        torch.randint(0, 93, (4,)),
        torch.randint(0, 2, (4,)),
    )
    out.sum().backward()
    enc = dict(model.encoder.named_parameters())
    assert float(enc["cross_attn.W_q.weight"].grad.abs().max()) > 0.0
    assert float(enc["cross_attn.W_k.weight"].grad.abs().max()) > 0.0


def test_encodings_no_longer_claims_no_roster_arm_uses_the_flat_variant():
    src = (ROOT / "src" / "models" / "encodings.py").read_text()
    assert "No Exp 2 roster arm" not in src, (
        "two transformer arms in results.xlsx use the flat degenerate variant"
    )


# ---------------------------------------------------------------------------
# Every published arm must be launchable.
# ---------------------------------------------------------------------------

def test_run_full_test_launches_every_published_arm():
    from src.utils.exp2_roster import EXP2_ARMS
    assert ("random_forest", "embeddings") in EXP2_ARMS
    assert ("random_forest", "onehot") in EXP2_ARMS
    assert ("transformer", "cross_attention") in EXP2_ARMS
    assert ("transformer", "film_attention") in EXP2_ARMS
    assert "experiment_manifest.env" in (ROOT / "scripts/run_full_test.sh").read_text()


# ---------------------------------------------------------------------------
# Seed spread is a SAMPLE standard deviation.
# ---------------------------------------------------------------------------

def test_seed_variance_uses_the_sample_divisor():
    from scripts.build_paper_results import _seed_variance, _aggregate_seeds

    vals = [0.050, 0.051, 0.053]
    rows = [
        {"model": "m", "variant": "v", "seed": s, "norm_mse_combined": x}
        for s, x in zip((26, 105, 115), vals)
    ]
    expected = float(pd.Series(vals).std(ddof=1))
    got = float(_seed_variance(rows).loc[0, "std"])
    # the sheet rounds to 8 decimals
    assert got == pytest.approx(expected, abs=1e-8)
    # ddof=0 would be ~18% smaller at n=3 -- the whole point of the fix.
    assert got > float(pd.Series(vals).std(ddof=0))

    agg = _aggregate_seeds(rows)
    assert float(agg.loc[0, "norm_mse_std"]) == pytest.approx(expected, abs=1e-8)


def test_single_run_arm_reports_zero_spread_not_nan():
    """Deterministic baselines have one run; NaN would poison the figures."""
    from scripts.build_paper_results import _seed_variance, _aggregate_seeds

    rows = [{"model": "moving_average", "variant": "none", "seed": 26,
             "norm_mse_combined": 0.0528}]
    assert float(_seed_variance(rows).loc[0, "std"]) == 0.0
    assert float(_aggregate_seeds(rows).loc[0, "norm_mse_std"]) == 0.0


# ---------------------------------------------------------------------------
# Analysis scripts must not mix real-unit and normalized predictions, and must
# not filter arms on the test metric.
# ---------------------------------------------------------------------------

_PRED_HEADER = ("State,Commodity,Import/Export,Time,"
                "pred_value,pred_weight,true_value,true_weight\n")


def _write_run(d, *, norm_scale, with_norm_file, predictions_file=None):
    d.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"Iowa,10 Cereals,Export,2025-0{i+1}-01,"
        f"{0.2*norm_scale},{0.2*norm_scale},{0.5*norm_scale},{0.5*norm_scale}\n"
        for i in range(5)
    )
    (d / "predictions.csv").write_text(_PRED_HEADER + rows)
    if with_norm_file:
        norm = "".join(
            f"Iowa,10 Cereals,Export,2025-0{i+1}-01,0.2,0.2,0.5,0.5\n"
            for i in range(5)
        )
        (d / "predictions_norm.csv").write_text(_PRED_HEADER + norm)
    payload = {"n_samples": 5}
    if predictions_file:
        payload["predictions_file"] = predictions_file
    (d / "metrics.json").write_text(json.dumps(payload))


def test_dm_loader_prefers_the_normalized_table(tmp_path):
    from ext.eval.diebold_mariano import normalized_predictions_path, load_run

    run = tmp_path / "s4_s4nd_2d_seed26"
    _write_run(run, norm_scale=1e7, with_norm_file=True,
               predictions_file="predictions_norm.csv")
    assert os.path.basename(normalized_predictions_path(run)) == "predictions_norm.csv"
    # 0.5*((0.2-0.5)^2 + (0.2-0.5)^2) = 0.09 -- a plausible normalized loss.
    assert all(v == pytest.approx(0.09) for v in load_run(run).values())


def test_dm_loader_refuses_real_unit_predictions(tmp_path):
    from ext.eval.diebold_mariano import load_run

    run = tmp_path / "s4_s4nd_2d_seed26"
    _write_run(run, norm_scale=1e7, with_norm_file=False)
    with pytest.raises(SystemExit, match="REAL units"):
        load_run(run)


def test_dm_family_labels_are_not_prefix_collisions():
    from ext.eval.diebold_mariano import family_of

    assert family_of("random_forest_onehot") == "RandomForest"
    assert family_of("random_walk_none") == "RandomWalk"
    assert family_of("mamba_nd_cross_attention_2d") == "Mamba-ND"
    assert family_of("mamba_embeddings") == "Mamba"
    assert family_of("moving_average_none") == "MovingAvg"
    assert family_of("s4_s4nd_2d_state") == "S4ND-2D"


def test_significance_loader_has_no_test_metric_filter():
    # Comment lines stripped: the fix deliberately quotes the old expression in
    # a comment explaining why it was wrong.
    code = "\n".join(
        line for line in (ROOT / "util" / "make_significance.py").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "np.mean(seed_means) > 0.5" not in code, (
        "dropping arms by their TEST loss conditions the omnibus test on "
        "performance; gate on predictions_file / unit scale instead"
    )
    assert "_SKIPPED" not in code


# ---------------------------------------------------------------------------
# S4ND grid loss must reduce over KEPT cells, not the padded lattice.
# ---------------------------------------------------------------------------

def test_selected_grid_targets_match_native_loss_and_gradients():
    from types import SimpleNamespace
    from src.training.losses import MaskedMSELoss
    from src.utils.s4_benchmark import select_valid_targets

    torch.manual_seed(0)
    mask = torch.tensor([[1] * 7 + [0] * 5] * 2, dtype=torch.bool)
    pred = torch.randn(2, 12, 2, requires_grad=True)
    target = torch.randn_like(pred)
    selected, truth = select_valid_targets(
        pred, target, SimpleNamespace(), {"valid_mask": mask})
    loss = torch.nn.functional.mse_loss(selected, truth)
    reference = MaskedMSELoss()(pred, target[..., 0], target[..., 1], mask)
    torch.testing.assert_close(loss, reference)
    a = torch.autograd.grad(loss, pred, retain_graph=True)[0]
    b = torch.autograd.grad(reference, pred)[0]
    torch.testing.assert_close(a, b)
    assert torch.count_nonzero(a[~mask]) == 0


def test_s4_eval_reports_smape_and_params():
    src = (ROOT / "scripts" / "s4_eval.py").read_text()
    assert 'compute_evaluation_metrics' in src, "s4/s4nd rows had a blank smape column"
    assert '"num_params"' in src, "s4/s4nd rows had a blank num_params column"
    assert '"predictions_file": "predictions_norm.csv"' in src


# ---------------------------------------------------------------------------
# Leakage: the evaluated cohort must be frozen on train data and must not move
# with the sweep axes.
# ---------------------------------------------------------------------------

def _tiny_grid():
    """A synthetic raw grid: 3 combos over 2009-01..2025-05.

    Combo A is dense throughout. Combo B is dense through train and thins out
    during 2024-25 -- the survivorship case. Combo C is sparse in train.
    """
    times = pd.date_range("2009-01-01", "2025-05-01", freq="MS")
    rows = []
    for state, kind in (("A", "dense"), ("B", "thins"), ("C", "sparse")):
        for i, t in enumerate(times):
            if kind == "dense":
                v = 100.0
            elif kind == "thins":
                # Dense through train, then goes quiet -- the exact shape the
                # full-timeline filter punishes and the train-only filter does not.
                v = 100.0 if t <= pd.Timestamp("2023-12-01") else 0.0
            else:
                v = 100.0 if i % 3 == 0 else 0.0
            rows.append({"State": state, "Commodity": "X", "Import/Export": "Export",
                         "Time": t, "Value": v, "Weight": v})
    return pd.DataFrame(rows)


def _cohort(grid, *, train_only, lag_count=12, input_len=36):
    from src.data.dataloader import TradeDataPipeline, DataConfig

    pipe = TradeDataPipeline(DataConfig(
        cohort_from_train_only=train_only, lag_count=lag_count, input_len=input_len,
    ))
    pipe.filter_sparse_combos(grid)
    return set(pipe.kept_combos)


def test_train_only_cohort_keeps_series_that_thin_during_test():
    """The whole survivorship argument, on a grid where the answer is known."""
    grid = _tiny_grid()
    train_only = _cohort(grid, train_only=True)
    with pytest.raises(ValueError, match="cohort_from_train_only must be True"):
        _cohort(grid, train_only=False)
    b = ("B", "X", "Export")
    assert b in train_only, "a series dense through train must be evaluated"



def test_cohort_is_invariant_to_lag_and_window():
    """Exp 3 sweeps input_len and lag_count over {6,12,24,36}.

    The cohort must not move with them, or each sweep cell evaluates a
    different population and the cells stop being comparable. This is why
    filter_sparse_combos measures over [start, train_end] rather than over the
    rows add_lags leaves behind.
    """
    grid = _tiny_grid()
    reference = _cohort(grid, train_only=True, lag_count=12, input_len=36)
    for lag in (6, 12, 24, 36):
        for win in (6, 12, 24, 36):
            got = _cohort(grid, train_only=True, lag_count=lag, input_len=win)
            assert got == reference, (
                f"cohort moved at lag_count={lag}, input_len={win}: the Exp 3 "
                f"sweep cells would evaluate different populations"
            )


def test_cohort_filter_refuses_an_empty_train_slice():
    """A misconfigured train_end must fail loudly, not select on everything."""
    from src.data.dataloader import TradeDataPipeline, DataConfig

    pipe = TradeDataPipeline(DataConfig(start="2008-01-01", train_end="2008-01-01"))
    with pytest.raises(ValueError, match="no grid month is <="):
        pipe.filter_sparse_combos(_tiny_grid())


# ---------------------------------------------------------------------------
# Leakage: no analysis or figure may rank on a best-of-N test draw.
# ---------------------------------------------------------------------------

def test_exp1_sheet_is_aggregated_across_seeds():
    src = (ROOT / "scripts" / "build_paper_results.py").read_text()
    assert '"Exp1_Aggregated": _aggregate_seeds(exp1_rows)' in src, (
        "a per-run sheet sorted by test MSE puts the best of 25 seeds on row 1, "
        "directly above single-run deterministic baselines"
    )
    assert '"Exp1_PerSeed"' in src, "the per-seed detail must remain available"


def test_exp1_figure_ranks_on_the_mean_not_the_best_run():
    src = (ROOT / "util" / "make_paper_figures.py").read_text()
    assert "def load_exp1_mean" in src
    assert "return {m: st.mean(xs) for m, xs in agg.items()}" in src, (
        "the Exp 1 rank chart must use the seed mean"
    )
    assert "return {m: min(" not in src and "else min(xs)" not in src, (
        "ranking Exp 1 models by their minimum over 25 seeds is selection on "
        "the test metric"
    )


# ---------------------------------------------------------------------------
# Leakage: the uncohorted tx.py population must not be reachable by accident.
# ---------------------------------------------------------------------------

def test_uncohorted_tx_datasets_refuse_without_explicit_opt_in():
    src = (ROOT / "external" / "s4" / "src" / "dataloaders" / "tx.py").read_text()
    assert "allow_uncohorted" in src, (
        "tx.py registers `s4_onehot`, which applies NO cohort filter, one "
        "character from the benchmark's `s4_onehot_u`"
    )
    assert "raise ValueError(" in src.split("def setup", 1)[1][:4000]


def test_benchmark_s4_onehot_experiment_points_at_the_cohorted_dataset():
    cfg = (ROOT / "external" / "s4" / "configs" / "experiment" / "s4_onehot.yaml").read_text()
    assert "s4_onehot_u" in cfg
    assert "_name_: s4_onehot\n" not in cfg, (
        "that name resolves to the uncohorted tx.py dataset"
    )


def test_no_analysis_script_defaults_to_a_machine_specific_repo():
    for rel in ("ext/eval/diebold_mariano.py", "ext/eval/s4_dump_preds.py"):
        src = (ROOT / rel).read_text()
        assert "/root/full" not in src and 'expanduser("~/full")' not in src, (
            f"{rel} silently finds zero runs outside one machine"
        )


# ---------------------------------------------------------------------------
# Leakage quarantine: nothing may turn a pre-correction artifact into a number.
# ---------------------------------------------------------------------------

# Every script that reads run artifacts and emits a reported number, table or
# paper figure. If a new one is added it belongs here.
GATED_CONSUMERS = [
    "scripts/collect_results.py",
    "scripts/build_paper_results.py",
    "util/make_significance.py",
    "util/make_mase_table.py",
    "util/make_cd_diagram.py",
    "util/make_condensed.py",
    "util/backfill_mase.py",
    "util/build_sweep_heatmaps.py",
    "ext/eval/diebold_mariano.py",
]


@pytest.mark.parametrize("rel", GATED_CONSUMERS)
def test_artifact_consumers_check_provenance(rel):
    src = (ROOT / rel).read_text()
    assert ("assert_corrected_runs" in src
            or "corrected_per_combo_artifact" in src
            or "validated_exp2_artifacts" in src
            or "load_model_losses" in src), (
        f"{rel} turns run artifacts into a reported number without checking "
        f"that they came from the corrected train-only cohort"
    )


def test_stale_payload_is_refused_by_the_shared_gate(tmp_path):
    from src.utils.provenance import assert_corrected_runs

    run = tmp_path / "lstm_embeddings_seed26"
    run.mkdir()
    # The exact shape of every artifact in outputs/full: no cohort block,
    # 4,515 targets (903 lanes x 5 months).
    (run / "metrics.json").write_text(json.dumps({
        "model": "lstm", "variant": "embeddings", "seed": 26,
        "mode": "one_step_normalized", "n_samples": 4515,
        "metrics": {"norm_mse_combined": 0.0507},
    }))
    with pytest.raises(RuntimeError, match="stale run"):
        assert_corrected_runs(tmp_path, what="a test")


def test_metadata_without_source_and_completion_cannot_pass_the_shared_gate(tmp_path):
    from src.utils.provenance import (
        assert_corrected_runs, cohort_provenance, data_protocol,
        CORRECTED_NUM_COMBOS, CORRECTED_TEST_MONTHS,
    )

    run = tmp_path / "moving_average_none_seed26"
    run.mkdir()
    header = ("State,Commodity,Import/Export,Time,"
              "pred_value,pred_weight,true_value,true_weight\n")
    rows = "".join(
        f"S{c},X,Export,2025-{m:02d}-01,0.2,0.2,0.5,0.5\n"
        for c in range(CORRECTED_NUM_COMBOS)
        for m in range(1, CORRECTED_TEST_MONTHS + 1)
    )
    (run / "predictions.csv").write_text(header + rows)
    (run / "metrics.json").write_text(json.dumps({
        "model": "moving_average", "variant": "none", "seed": 26,
        "mode": "one_step_normalized",
        "n_samples": CORRECTED_NUM_COMBOS * CORRECTED_TEST_MONTHS,
        "predictions_file": "predictions.csv",
        "cohort": cohort_provenance(CORRECTED_NUM_COMBOS, True),
        # The contract also pins the training-history settings that decide
        # which supervised rows exist, so an artifact cannot be compared
        # against one trained on a different window/lag budget.
        "data_protocol": data_protocol(input_len=36, lag_count=12,
                                       train_target_start=None),
        "metrics": {"norm_mse_combined": 0.0528},
    }))
    with pytest.raises(RuntimeError, match="Refusing"):
        assert_corrected_runs(tmp_path, what="a test")


def test_dm_comparator_name_matches_the_run_directory_convention():
    """BASELINE_ARM must be the arm key, or the script can never run at all."""
    from ext.eval.diebold_mariano import BASELINE_ARM

    assert BASELINE_ARM == "moving_average_none", (
        "arm keys are the run directory minus _seed<N>; the bare "
        "'moving_average' never matched and masked the missing provenance gate"
    )


# ---------------------------------------------------------------------------
# PR #20 review follow-ups.
# ---------------------------------------------------------------------------

def test_every_launcher_call_propagates_failure():
    """`set -uo pipefail` without -e means a failed arm is just a loop exit
    status. Without `|| FAILED=1` the sweep prints "complete" while a published
    arm has no valid artifact."""
    import re

    sh = (ROOT / "scripts" / "run_full_test.sh").read_text()
    calls = [
        (i + 1, line.strip())
        for i, line in enumerate(sh.splitlines())
        if re.match(r"\s*(run_github|run_baseline|run_s4) ", line) and "()" not in line
    ]
    assert calls, "no launcher calls found -- did the script move?"
    missing = [c for c in calls if "|| FAILED=1" not in c[1]]
    assert not missing, f"launcher calls that swallow failure: {missing}"


def test_dm_refuses_predictions_without_provenance(tmp_path, monkeypatch):
    """ext/eval/s4_dump_preds.py writes predictions.csv with no metrics.json.
    A conditional provenance check let such a directory into the analysis
    unverified, and a legacy 903-lane dump would then shrink the all-arm key
    intersection back to the survivorship-biased cohort."""
    import ext.eval.diebold_mariano as dm

    run = tmp_path / "s4_s4nd_2d_seed26"
    run.mkdir()
    (run / "predictions.csv").write_text(
        "State,Commodity,Import/Export,Time,"
        "pred_value,pred_weight,true_value,true_weight\n"
        "Iowa,10 Cereals,Export,2025-01-01,0.2,0.2,0.5,0.5\n"
    )
    monkeypatch.setattr(dm, "FULL_DIR", str(tmp_path))
    monkeypatch.setattr(dm, "S4_PRED_DIR", str(tmp_path / "_absent"))
    with pytest.raises(SystemExit, match="no metrics.json"):
        dm.load_arms()


def test_dm_refuses_truncated_declared_predictions(tmp_path, monkeypatch):
    """Current-looking metrics must not let an incomplete prediction table
    shrink the DM sample through the later key intersection."""
    from src.utils.provenance import cohort_provenance, data_protocol
    import ext.eval.diebold_mariano as dm

    run = tmp_path / "gru_embeddings_seed26"
    run.mkdir()
    (run / "predictions.csv").write_text(
        "State,Commodity,Import/Export,Time,"
        "pred_value,pred_weight,true_value,true_weight,"
        "pred_value_real,pred_weight_real,true_value_real,true_weight_real\n"
        "Iowa,10 Cereals,Export,2025-01-01,0.2,0.2,0.5,0.5,2,2,5,5\n"
    )
    (run / "metrics.json").write_text(json.dumps({
        "model": "gru", "variant": "embeddings", "seed": 26,
        "mode": "one_step_normalized", "n_samples": 4590,
        "prediction_units": "normalized", "scale_dependent_metric_units": "normalized",
        "percentage_metric_units": "real",
        "predictions_file": "predictions.csv",
        "cohort": cohort_provenance(918, True),
        "data_protocol": data_protocol(36, 12, None),
        "metrics": {"norm_mse_combined": 0.1},
    }))
    monkeypatch.setattr(dm, "FULL_DIR", str(tmp_path))
    monkeypatch.setattr(dm, "S4_PRED_DIR", str(tmp_path / "_absent"))
    monkeypatch.setattr(dm, "corrected_per_combo_artifact", lambda payload: (True, ""))
    with pytest.raises(SystemExit, match="has 1 rows; expected 4590"):
        dm.load_arms()


def test_condensed_builder_reads_the_per_seed_exp1_sheet():
    """condense() aggregates with n_seeds=("seed", "nunique"), and
    Exp1_Aggregated no longer carries a `seed` column."""
    src = (ROOT / "util" / "make_condensed.py").read_text()
    assert 'condense(xl.parse("Exp1_Aggregated")' not in src, (
        "aggregated sheet has no `seed` column -- condense() raises KeyError"
    )
    assert "_exp1_rows(ROOT)" in src
    assert "pd.ExcelFile" not in src
