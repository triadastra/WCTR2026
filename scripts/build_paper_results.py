#!/usr/bin/env python
"""Build the final paper results.xlsx from every experiment's TEST metrics.

This is the SINGLE source of the paper's metric tables. It reads each run's
metrics.json (written by inference.py / run_baseline.py / s4_eval.py over the
TEST split — never val; val is selection-only) and emits one workbook with the
FULL loss suite per model, one sheet per experiment.

Driving loss (training + model selection, uniform across all models):
    norm_mse_combined  (per-combo MinMax-normalized MSE).
Every other column is REPORTED but never used to select a model.

Experiments (see scripts/run_all_experiments.sh):
    Exp1_Aggregated     outputs/agg/<model>_<variant>_seed<seed>/metrics.json
    Exp2_PerCombo       outputs/full/<model>_<variant>_seed<seed>/metrics.json   (headline)
    Exp3_Sweep          outputs/sweep/<run>/metrics.json
    Exp2_SeedVariance   pivot of Exp2 across seeds -> mean/std of the driving loss

Usage:
    python scripts/build_paper_results.py --repo . --out results.xlsx
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.provenance import (
    BENCHMARK_PROTOCOL_VERSION,
    CORRECTED_TEST_DATES,
    CORRECTED_TEST_MONTHS,
    corrected_aggregate_artifact,
    corrected_per_combo_artifact,
    corrected_prediction_artifact,
    prediction_truth_digest,
)
from src.utils.experiment_manifest import (
    MANIFEST_RELATIVE_PATH,
    exp1_expected_runs,
    exp1_protocol,
    exp1_source_domain,
    load_manifest,
)
from src.utils.run_contract import expected_exp1_run_contract
from src.utils.exp3_summary import summarize_sweep, sweep_summary_text
from src.utils.report_metadata import (
    attach_report_evidence, reporting_sheets, verify_report_snapshot, metric_dictionary,
    verify_workbook,
)
from src.inference.metrics import (
    REQUIRED_EVALUATION_METRICS,
    compute_evaluation_metrics,
)

RUN_RE = re.compile(r"^(?P<model>.+?)_(?P<variant>[^_]+)_seed(?P<seed>\d+)$")

# Exp 3 is a window x lag sensitivity grid: the manifest fixes its models,
# encoder, seeds and cells. Everything else under outputs/sweep is a
# stray, and they were all being ingested as results -- a smoke test
# (s4smoke_embeddings_w12_l6_seed999), off-protocol seeds (42), and single
# half-finished onehot cells. The smoke run's metrics.json records model="s4",
# so the payload cannot be trusted to exclude it; match the directory name,
# which is the only place the run's real identity survives.
# Three pinned seeds, matching both canonical sweep launchers. Partial coverage
# is rejected below rather than averaged with a different seed count in only a
# subset of cells.
SWEEP_RE = re.compile(
    r"^(?P<model>lstm|transformer|s4|s4_bidir)_(?P<variant>embeddings)"
    r"_w(?P<window>\d+)_l(?P<lag>\d+)_seed(?P<seed>26|105|115)$"
)

# Driving loss first, then the single canonical reporting suite shared by the
# evaluators, provenance gate, collector, and paper builder. Parameter counts
# are optional because closed-form baselines and native tree bundles do not
# have a meaningful torch parameter count.
COLS = [*REQUIRED_EVALUATION_METRICS, "num_params"]


def _row_from_payload(payload, run_name, *, run_kind=None):
    flat = payload.get("metrics", payload)
    row = {
        "model": payload["model"],
        "variant": payload["variant"],
        "seed": payload["seed"],
        "mode": payload.get("mode", ""),
    }
    if run_kind is not None:
        row["run_kind"] = run_kind
    for column in COLS:
        if column not in flat:
            continue
        try:
            value = float(flat[column])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Refusing to publish {run_name}: metric {column} is not numeric"
            ) from exc
        if not math.isfinite(value):
            raise RuntimeError(
                f"Refusing to publish {run_name}: metric {column} is not finite"
            )
        row[column] = value
    return row


def _validate_exp1_contract(repo, payload, expected_run, protocol):
    source_fingerprint = payload.get("source_fingerprint")
    if source_fingerprint != protocol["source_fingerprint"]:
        raise RuntimeError(
            f"Refusing to publish {expected_run.directory_name}: source fingerprint "
            f"{source_fingerprint!r}, expected {protocol['source_fingerprint']!r}"
        )
    source_domain = payload.get("source_domain")
    canonical_domain = exp1_source_domain(load_manifest(repo / MANIFEST_RELATIVE_PATH))
    if source_domain != canonical_domain:
        raise RuntimeError(
            f"Refusing to publish {expected_run.directory_name}: source state/calendar "
            f"domain does not match the canonical eight-state protocol"
        )
    contract = payload.get("run_contract")
    expected_contract = expected_exp1_run_contract(
        repo, expected_run, protocol, canonical_domain
    )
    if contract != expected_contract:
        raise RuntimeError(
            f"Refusing to publish {expected_run.directory_name}: embedded run contract "
            "does not match the current model/data/training/selection manifest"
        )
    if expected_run.kind == "baseline":
        expected_window = (
            protocol["moving_average_window"]
            if expected_run.model == "moving_average" else None
        )
        if payload.get("window") != expected_window:
            raise RuntimeError(
                f"Refusing to publish {expected_run.directory_name}: baseline window="
                f"{payload.get('window')!r}, expected {expected_window!r}"
            )


def _exp1_rows(repo: Path):
    """Load exactly the canonical aggregate arm/seed population.

    This is intentionally stricter than the generic result reader.  Experiment
    1's former collector scanned every directory and trusted payload labels,
    which made smoke runs, old 3-seed artifacts, duplicate baseline labels, and
    even non-aggregate one-series runs publishable as paper results.
    """
    runs_dir = repo / "outputs" / "agg"
    if not runs_dir.is_dir():
        raise RuntimeError(f"Exp 1 results directory is missing: {runs_dir}")
    manifest = load_manifest(repo / MANIFEST_RELATIVE_PATH)
    expected_runs = exp1_expected_runs(manifest)
    expected = {run.directory_name: run for run in expected_runs}
    discovered = {
        directory.name: directory
        for directory in runs_dir.iterdir()
        if directory.is_dir() and (directory / "metrics.json").is_file()
    }
    missing = sorted(set(expected).difference(discovered))
    unexpected = sorted(set(discovered).difference(expected))
    if missing or unexpected:
        details = []
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            details.append(f"missing {len(missing)} canonical run(s): {shown}")
        if unexpected:
            shown = ", ".join(unexpected[:5]) + (" ..." if len(unexpected) > 5 else "")
            details.append(f"unexpected {len(unexpected)} run(s): {shown}")
        raise RuntimeError(
            "Refusing to publish an inexact Exp 1 manifest (" + "; ".join(details) + ")"
        )

    protocol = exp1_protocol(manifest)
    rows = []
    truth_digest = None
    source_truth = None
    for name, expected_run in sorted(expected.items()):
        metrics_path = discovered[name] / "metrics.json"
        try:
            payload = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Refusing to publish {name}: unreadable metrics ({exc})") from exc
        identity = {
            "model": expected_run.model,
            "variant": expected_run.variant,
            "seed": expected_run.seed,
        }
        for key, wanted in identity.items():
            actual = payload.get(key)
            if actual != wanted or type(actual) is not type(wanted):
                raise RuntimeError(
                    f"Refusing to publish {name}: payload {key}={actual!r}, "
                    f"directory/manifest requires {wanted!r}"
                )
        current, reason = corrected_aggregate_artifact(payload)
        if not current:
            raise RuntimeError(f"Refusing to publish stale Exp 1 run {name}: {reason}")
        _validate_exp1_contract(repo, payload, expected_run, protocol)
        from src.utils.exp1_completion import validate_exp1_completion_artifact
        try:
            validate_exp1_completion_artifact(payload, metrics_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Refusing to publish incomplete Exp 1 run {name}: {exc}") from exc
        actual_protocol = payload.get("data_protocol", {})
        for key in ("input_len", "lag_count", "train_target_start"):
            if actual_protocol.get(key) != protocol[key]:
                raise RuntimeError(
                    f"Refusing to publish {name}: data_protocol.{key}="
                    f"{actual_protocol.get(key)!r}, expected {protocol[key]!r}"
                )
        current, reason = corrected_prediction_artifact(
            payload,
            metrics_path,
            expected_num_combos=1,
            expected_months=CORRECTED_TEST_MONTHS,
            expected_dates=CORRECTED_TEST_DATES,
            expected_combos={("ALL", "ALL", "ALL")},
        )
        if not current:
            raise RuntimeError(f"Refusing to publish incomplete Exp 1 run {name}: {reason}")
        from src.utils.exp1_truth import load_exp1_truth, validate_exp1_truth
        if source_truth is None:
            source_truth = load_exp1_truth(repo, Path(os.environ.get("DATA_DIR", str(repo))))
        current, reason = validate_exp1_truth(payload, metrics_path, source_truth)
        if not current:
            raise RuntimeError(f"Refusing to publish Exp 1 run {name}: {reason}")
        this_digest = prediction_truth_digest(payload, metrics_path)
        if truth_digest is None:
            truth_digest = this_digest
        elif this_digest != truth_digest:
            raise RuntimeError(
                f"Refusing to publish {name}: test targets differ from another "
                "canonical Exp 1 run (mixed data revisions or normalization stats)"
            )
        rows.append(attach_report_evidence(
            _row_from_payload(payload, name, run_kind=expected_run.kind), payload, metrics_path))
    print(
        f"[exp1] validated {len(rows)} runs across "
        f"{len({(row['model'], row['variant']) for row in rows})} arms; "
        "all share one aggregate test-target fingerprint"
    )
    return rows, manifest


def _exp2_rows(repo: Path):
    """Use the same complete validation as every statistical consumer."""
    from src.utils.exp2_artifacts import validated_exp2_artifacts
    return [attach_report_evidence(_metric_row(payload, path.parent), payload, path)
            for _, path, payload in validated_exp2_artifacts(repo)]


def _metric_row(payload, directory):
    """Format an already loaded payload without reopening mutable run files."""
    flat = payload.get("metrics", payload)
    match = RUN_RE.match(directory.name)
    row = {
        "model": payload.get("model", match.group("model") if match else directory.name),
        "variant": payload.get("variant", match.group("variant") if match else ""),
        "seed": payload.get("seed", int(match.group("seed")) if match else None),
        "mode": payload.get("mode", ""),
    }
    arch = payload.get("architecture") or {}
    if arch.get("bidirectional") is not None:
        row["bidirectional"] = bool(arch["bidirectional"])
    for column in COLS:
        if column in flat:
            row[column] = float(flat[column])
    return row


def _rows_from(runs_dir: Path, require_corrected=False):
    """One row per run dir containing a metrics.json."""
    rows = []
    if not runs_dir.is_dir():
        return rows
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        payload = json.loads(mj.read_text())
        if require_corrected:
            current, reason = corrected_per_combo_artifact(payload)
            if not current:
                raise RuntimeError(
                    f"Refusing to publish stale run {d.name} from {runs_dir}: "
                    f"{reason}. Rerun it under the corrected train-only cohort."
                )
            current, reason = corrected_prediction_artifact(payload, mj, verify_scaled_metrics=True)
            if not current:
                raise RuntimeError(
                    f"Refusing to publish incomplete run {d.name} from {runs_dir}: "
                    f"{reason}. Rerun its corrected evaluation."
                )
        rows.append(_metric_row(payload, d))
    return rows


def _sweep_rows(runs_dir: Path):
    """Exp 3 cells, keyed by the grid coordinates the experiment actually varies.

    Carries window/lag_count as columns. Without them the sheet is sixteen
    indistinguishable rows per model -- the coordinates ARE the experiment, and
    their absence is why the stray runs went unnoticed for so long.
    """
    rows, skipped, stale = [], [], []
    if not runs_dir.is_dir():
        raise RuntimeError(f"Exp 3 sweep incomplete: results directory is missing: {runs_dir}")
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        m = SWEEP_RE.match(d.name)
        if not m:
            skipped.append(d.name)
            continue
        payload = json.loads(mj.read_text())
        current, reason = corrected_per_combo_artifact(payload)
        if not current:
            stale.append(f"{d.name} ({reason})")
            continue
        current, reason = corrected_prediction_artifact(payload, mj, verify_scaled_metrics=True)
        if not current:
            stale.append(f"{d.name} ({reason})")
            continue
        from src.utils.exp3_protocol import validate_contract, run_name
        if d.name != run_name(m.group("model"), int(m.group("window")),
                              int(m.group("lag")), int(m.group("seed"))):
            stale.append(f"{d.name} (noncanonical coordinate spelling)")
            continue
        current, reason = validate_contract(payload, m.group("model"),
                                            int(m.group("window")), int(m.group("lag")),
                                            int(m.group("seed")))
        if not current:
            stale.append(f"{d.name} ({reason})")
            continue
        from src.utils.training_completion import validate_completion_artifact
        try:
            validate_completion_artifact(payload, mj)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            stale.append(f"{d.name} ({exc})")
            continue
        protocol = payload["data_protocol"]
        expected_protocol = {
            "input_len": int(m.group("window")),
            "lag_count": int(m.group("lag")),
            "train_target_start": "2015-01-01",
        }
        if any(protocol.get(k) != v for k, v in expected_protocol.items()):
            stale.append(f"{d.name} (training-history protocol mismatch)")
            continue
        flat = payload.get("metrics", payload)
        row = {
            "model": m.group("model"),
            "variant": m.group("variant"),
            "seed": int(m.group("seed")),
            "mode": payload.get("mode", ""),
            "window": int(m.group("window")),
            "lag_count": int(m.group("lag")),
        }
        for c in COLS:
            if c in flat:
                row[c] = float(flat[c])
        rows.append(attach_report_evidence(row, payload, mj))
    rows.sort(key=lambda r: (r["model"], r["window"], r["lag_count"], r["seed"]))
    if skipped:
        shown = ", ".join(sorted(skipped)[:6]) + (" ..." if len(skipped) > 6 else "")
        print(f"[exp3] excluded {len(skipped)} non-canonical run(s): {shown}")
    if stale:
        shown = ", ".join(sorted(stale)[:4])
        raise RuntimeError(
            f"Refusing to publish {len(stale)} stale canonical Exp 3 run(s): "
            f"{shown}. Rerun the corrected sweep."
        )
    # Report per-cell seed coverage rather than a bare row count: a partially
    # finished sweep otherwise looks the same as a complete one, and uneven
    # coverage across cells is exactly the defect that corrupted the heatmaps.
    cov = defaultdict(set)
    for r in rows:
        cov[(r["model"], r["window"], r["lag_count"])].add(r["seed"])
    from src.utils.exp3_protocol import coordinates
    expected = defaultdict(set)
    for model, window, lag, seed in coordinates():
        expected[(model, window, lag)].add(seed)
    sizes = sorted({len(v) for v in cov.values()})
    print(f"[exp3] {len(rows)} rows over {len(cov)} cells (expect {len(expected)} from manifest); "
          f"seeds per cell: {sizes}")
    if cov != expected:
        ragged = [k for k in cov.keys() | expected.keys() if cov.get(k) != expected.get(k)]
        raise RuntimeError(
            f"Exp 3 sweep incomplete: {len(cov)}/{len(expected)} cells, "
            f"seed counts {sizes}; examples {sorted(ragged)[:3]}"
        )
    return rows


def _df(rows, sort=True, extra=()):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    present = [c for c in COLS if c in df.columns]
    lead = [c for c in ("run_kind", "bidirectional", *extra) if c in df.columns]
    df = df[["model", "variant", "seed", "mode"] + lead + present]
    if sort and "norm_mse_combined" in df.columns:
        df = df.sort_values("norm_mse_combined", na_position="last").reset_index(drop=True)
    return df


def _seed_variance(per_combo_rows):
    """mean/std of the driving loss across seeds, per (model, variant)."""
    agg = defaultdict(dict)
    for r in per_combo_rows:
        if r.get("norm_mse_combined") is None:
            continue
        agg[(r["model"], r["variant"])][f"seed{r['seed']}"] = r["norm_mse_combined"]
    out = []
    for (model, variant), seeds in sorted(agg.items()):
        vals = list(seeds.values())
        row = {"model": model, "variant": variant, **seeds}
        if vals:
            s = pd.Series(vals)
            row["mean"] = float(s.mean())
            row["std"] = float(s.std(ddof=1)) if len(vals) > 1 else 0.0
            row["n_seeds"] = len(vals)
        out.append(row)
    return pd.DataFrame(out)


def _aggregate_seeds(rows):
    """Per (model, variant): mean of every reported loss across seeds, + std of
    the DRIVING loss (norm_mse_combined) + n_seeds. This is the headline table."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    metric_cols = [c for c in REQUIRED_EVALUATION_METRICS if c in df.columns]
    g = df.groupby(["model", "variant"], sort=False)
    out = g[metric_cols].mean()
    if "norm_mse_combined" in df.columns:
        # Seeds are a sample from the initialization/training distribution, so
        # report sample spread.  The previous population divisor understated
        # the already-large Exp 1 uncertainty; one deterministic baseline run
        # has no sampling spread and is explicitly shown as zero.
        out["norm_mse_std"] = (
            g["norm_mse_combined"].std(ddof=1).fillna(0.0)
        )
    out["n_seeds"] = g["seed"].nunique().astype(int)
    out = out.reset_index()
    for metadata in ("run_kind", "bidirectional", "num_params"):
        if metadata in df.columns:
            if g[metadata].nunique(dropna=False).gt(1).any():
                raise RuntimeError(f"Seed architecture/identity mismatch: {metadata}")
            values = g[metadata].first().rename(metadata).reset_index()
            out = out.merge(values, on=["model", "variant"], validate="one_to_one")
    if "norm_mse_combined" in out.columns:
        out = out.sort_values("norm_mse_combined", na_position="last").reset_index(drop=True)
    return out


def _ensemble(repo, full_dir):
    """Score every canonical seed, joined by exact target identity; no omissions."""
    from src.utils.exp2_artifacts import validated_exp2_artifacts
    from src.utils.scaled_metric_validation import source_scaled_metrics
    # outputs/ may be a symlink to separate storage, so the resolved run
    # directory's grandparent is not the repo that holds the manifest/configs.
    full_dir = Path(full_dir).resolve()
    artifacts = validated_exp2_artifacts(repo, runs_dir=full_dir)
    key = ["State", "Commodity", "Import/Export", "Time"]
    numeric = ["pred_value", "pred_weight", "true_value", "true_weight",
               "pred_value_real", "pred_weight_real", "true_value_real", "true_weight_real"]
    groups = defaultdict(list)
    for run, path, payload in artifacts:
        frame = pd.read_csv(path.parent / payload["predictions_file"],
                            dtype={c: str for c in key})
        if frame[key].isna().any().any() or frame.duplicated(key).any():
            raise RuntimeError(f"[ensemble] invalid/duplicate target identities: {path}")
        groups[(run.model, run.variant)].append((run.seed, payload, frame.set_index(key).sort_index()))
    rows = []
    for (model, variant), members in sorted(groups.items()):
        seeds = [seed for seed, _, _ in members]
        if len(seeds) != len(set(seeds)) or len(seeds) < 2:
            raise RuntimeError(f"[ensemble] incomplete/duplicate seeds: {model}/{variant}")
        reference = members[0][2]
        truth_columns = [c for c in numeric if c.startswith("true_")]
        for _, _, frame in members:
            if not frame.index.equals(reference.index):
                raise RuntimeError(f"[ensemble] target populations differ: {model}/{variant}")
            if not frame[truth_columns].equals(reference[truth_columns]):
                raise RuntimeError(f"[ensemble] source truth differs: {model}/{variant}")
        ensemble = reference.copy()
        for column in [c for c in numeric if c.startswith("pred_")]:
            ensemble[column] = sum(frame[column] for _, _, frame in members) / len(members)
        pred = ensemble[["pred_value", "pred_weight"]].to_numpy()
        truth = ensemble[["true_value", "true_weight"]].to_numpy()
        pred_real = ensemble[["pred_value_real", "pred_weight_real"]].to_numpy()
        truth_real = ensemble[["true_value_real", "true_weight_real"]].to_numpy()
        metrics = compute_evaluation_metrics(pred, truth, pred_real, truth_real)
        identities = list(ensemble.index)
        metrics.update(source_scaled_metrics(
            members[0][1], pred, truth, [i[:3] for i in identities], identities=identities,
            predictions_real=pred_real, targets_real=truth_real))
        if set(REQUIRED_EVALUATION_METRICS).difference(metrics) or not all(
            math.isfinite(float(metrics[k])) for k in REQUIRED_EVALUATION_METRICS
        ):
            raise RuntimeError(f"[ensemble] incomplete/nonfinite metrics: {model}/{variant}")
        rows.append(dict(model=model, variant=variant, n_members=len(members),
                         n_targets=len(ensemble), seeds=",".join(map(str, sorted(seeds))), **metrics))
    return pd.DataFrame(rows).sort_values("norm_mse_combined").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results.xlsx")
    ap.add_argument("--claim", choices=("retrospective", "real-time", "confirmatory"),
                    default="retrospective", help="Current source supports retrospective evaluation only")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    if args.claim != "retrospective":
        raise RuntimeError(
            "Cannot certify this benchmark as real-time or confirmatory: the test "
            "period was previously inspected and the source has no audited "
            "per-observation release/vintage history. An untouched holdout and "
            "inputs bound to their actual availability dates are required."
        )

    exp1_rows, manifest = _exp1_rows(repo)
    exp2_rows = _exp2_rows(repo)
    sweep_rows = _sweep_rows(repo / "outputs/sweep")
    sweep_summary, sweep_marginals = summarize_sweep(sweep_rows, manifest)
    sheets = {
        # Headline means across seeds.  The old sheet contained 403 individual
        # rows sorted by TEST MSE, which put the luckiest of 25 stochastic
        # draws above one-shot baselines: an implicit best-of-25 selection on
        # the test set.  Raw runs remain available in Exp1_PerSeed.
        "Exp1_Aggregated": _aggregate_seeds(exp1_rows),
        "Exp1_PerSeed": _df(exp1_rows),
        "Exp1_SeedVariance": _seed_variance(exp1_rows),
        "Exp2_PerCombo": _aggregate_seeds(exp2_rows),       # headline: multi-seed mean±std
        "Exp2_PerSeed": _df(exp2_rows),                     # raw per-seed rows (transparency)
        "Exp2_Ensemble": _ensemble(repo, repo / "outputs/full"),  # prediction-averaged ensemble
        "Exp3_Sweep": _df(sweep_rows, sort=False,
                          extra=("window", "lag_count")),
        "Exp3_Summary": pd.DataFrame(sweep_summary),
        "Exp3_Marginals": pd.DataFrame(sweep_marginals),
        "Exp2_SeedVariance": _seed_variance(exp2_rows),
    }
    all_rows = [*exp1_rows, *exp2_rows, *sweep_rows]
    sheets.update(reporting_sheets({"Exp1": exp1_rows, "Exp2": exp2_rows, "Exp3": sweep_rows}))
    sheets["Metric_Definitions"] = metric_dictionary()
    sheets["Evaluation_Scope"] = pd.DataFrame([
        ("evaluation", "retrospective rolling one-step; fixed fitted model"),
        ("preprocessing", "canonical contracts require train-only cohort, mappings and scaling"),
        ("history", "inputs end at target month minus one; previous test outcomes become history"),
        ("selection", "validation checkpoint selection; test rankings and sweep minima are descriptive"),
        ("test_holdout", "previously inspected during development; not untouched"),
        ("real_time_availability", "unverified: source has no per-observation availability/vintage history"),
        ("leakage_free_claim", "NOT certified without qualification; no confirmed real-time claim"),
        ("release_schedule", "https://www.census.gov/foreign-trade/schedule.html"),
        ("revisions", "https://www.census.gov/foreign-trade/guide/revisions.html"),
    ], columns=["property", "status"])
    exp1_runs = len(exp1_rows)
    exp1_arms = len({(row["model"], row["variant"]) for row in exp1_rows})
    exp1_protocol_values = exp1_protocol(manifest)
    readme = pd.DataFrame({
        "key": ["scale", "split", "protocol", "driving_loss", "selection",
                "note", "Exp1 provenance", "Exp2 provenance", "Exp3 provenance",
                "Exp3 observations", "Target definition", "Source ambiguity",
                "Evaluation limitations"],
        "value": [
            "Train-fitted per-series MinMax; held-out values can fall outside 0-1. See Metric_Definitions.",
            "TEST only (2025-01..05): Exp 1 has 5 aggregate targets per run; "
            "Exp 2/3 have 4,590 lane-month targets per run. Val is selection-only.",
            "1-step walk-forward (h=1) at every test point, real history",
            "norm_mse_combined (MSE on normalized values)",
            "early-stopping on VAL norm-MSE; all other metrics reported not selected",
            "S4ND grid metrics masked to the 918 train-only kept combos (s4_eval.py)",
            f"Validated {exp1_runs} canonical runs across {exp1_arms} aggregate-series arms "
            f"under {BENCHMARK_PROTOCOL_VERSION}: trainable neural/SSM and tree arms "
            "use the same fixed 25-seed roster; 3 closed-form baselines run once at seed "
            f"{manifest['PAPER_EXP1_BASELINE_SEED']}. The eight-state aggregate sums the complete "
            "monthly raw panel BEFORE sparse-lane filtering; it is not a sum of Exp 2's "
            f"918-lane cohort. input_len={exp1_protocol_values['input_len']}, "
            f"lag_count={exp1_protocol_values['lag_count']}, "
            f"train_target_start={exp1_protocol_values['train_target_start']}, "
            f"training batch={exp1_protocol_values['batch_size']}. Exp1_Aggregated is the "
            "seed-mean headline; Exp1_PerSeed preserves every draw. All runs passed explicit "
            "aggregate identity, exact-manifest, prediction/MSE, and common-target checks.",
            "Corrected rerun protocol: 57 arms x seeds 26/105/115. Flat arms batch 918 "
            "(121,176 train samples); combo arms batch 1 (132 month-samples, each already all "
            "918 combos). Both land on 132 steps/epoch at 918 targets per gradient step. The workbook builder "
            "rejects pre-correction artifacts (903-series, full-timeline cohort) until this "
            "rerun is complete.",
            "Window x lag sensitivity grid, seeds 26/105/115, embeddings only, 64 configurations "
            f"(4 model/direction arms x 16) and 192 runs. Batch {manifest['PAPER_SWEEP_BATCH']} and "
            f"first training target {manifest['PAPER_SWEEP_TRAIN_START']} are uniform "
            "across all cells, so window/lag settings see the same labels and update opportunities. "
            "Off-protocol seeds and smoke-test runs are excluded by directory name. "
            "Every required cell and seed passed source, contract, checkpoint-completion "
            "and prediction validation. Exp3_Summary and Exp3_Marginals are recomputed "
            "from these same full-precision metric rows.",
            sweep_summary_text(sweep_summary),
            f"Pinned target measure: {manifest['PAPER_TARGET_MEASURE']}. "
            "For legacy_mixed, imports use Vessel Value ($US) / Vessel SWT (kg); "
            "exports use Containerized Vessel Total Exports Value ($US) / "
            "Containerized Vessel Total Exports SWT (kg). This is a mixed-scope "
            "trade-value/weight target, not uniform container demand or container counts. "
            "Blank selected fields and absent monthly observations are historically "
            "encoded as zero; source missingness is not independently resolved.",
            f"Source fingerprint {manifest['PAPER_SOURCE_FINGERPRINT']}; "
            f"duplicate policy {manifest['PAPER_DUPLICATE_POLICY']}; "
            f"report-grain audit {exp1_source_domain(manifest)['duplicate_audit']}. "
            "Hash and prediction validation establish reproducibility on the pinned "
            "snapshot, not that ambiguous duplicate rows are substantively additive.",
            "The test period has previously been inspected during development. "
            "Results are retrospective; seeds quantify optimization variability, "
            "not additional forecast origins. States can share economic shocks. "
            "Publication delays and release vintages are unverified, so this rolling "
            "one-step backtest with observed history does not establish real-time "
            "availability or a fixed-origin multi-step forecast. A frozen protocol "
            "and an untouched holdout are required for a fresh confirmatory claim.",
        ],
    })

    index_rows = [
        ("README", "All", "Metric and protocol provenance for this exact build."),
        ("Exp1_Aggregated", "Exp 1", "HEADLINE: one row per arm, metrics averaged over seeds."),
        ("Exp1_PerSeed", "Exp 1", "Raw canonical run rows; never rank by the first/best row."),
        ("Exp1_SeedVariance", "Exp 1", "Per-arm seed values, sample mean/std, and seed count."),
        ("Exp2_PerCombo", "Exp 2", "HEADLINE: one row per arm, metrics averaged over seeds."),
        ("Exp2_PerSeed", "Exp 2", "Raw canonical run rows."),
        ("Exp2_Ensemble", "Exp 2", "Prediction-averaged seed ensemble."),
        ("Exp2_SeedVariance", "Exp 2", "Per-arm seed values and sample spread."),
        ("Exp3_Sweep", "Exp 3", "Window x lag sensitivity cells."),
        ("Exp3_Summary", "Exp 3", "Computed cell ranges, seed SDs and observed test minima; descriptive only."),
        ("Exp3_Marginals", "Exp 3", "Window/lag marginal normalized MSE over the other axis and all seeds."),
    ]
    index_rows.extend([
        ("Run_Details", "All", "Per-run parameter count, validation-selected checkpoint and completion values."),
        ("Run_Values", "All", "All recorded evaluation/config fields; auxiliary values labelled by source."),
        ("Report_Files", "All", "Paths and SHA-256 of inputs used in this report; checked before publication."),
        ("Metric_Definitions", "All", "Units, weighting and seasonal scale definitions."),
        ("Model_Tensors", "All", "Measured parameter names, shapes, counts, dtypes and trainability when recorded."),
        ("Evaluation_Scope", "All", "Retrospective scope and unresolved holdout/release-availability limits."),
    ])
    index = pd.DataFrame(index_rows, columns=["sheet", "experiment", "contents"])

    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    try:
        with pd.ExcelWriter(temporary, engine="openpyxl") as xl:
            index.to_excel(xl, sheet_name="INDEX", index=False)
            readme.to_excel(xl, sheet_name="README", index=False)
            for name, df in sheets.items():
                frame = df if not df.empty else pd.DataFrame({"(not recorded)": []})
                # Large inventories can exceed Excel's row limit. Split only
                # at row boundaries, preserving every recorded value.
                for offset in range(0, max(1, len(frame)), 1_000_000):
                    sheet_name = name if offset == 0 else f"{name}_{offset // 1_000_000 + 1}"
                    frame.iloc[offset:offset + 1_000_000].to_excel(xl, sheet_name=sheet_name, index=False)
            # Formatting changes display only; stored numbers retain precision.
            for ws in xl.book.worksheets:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for cells in ws.iter_rows():
                    for cell in cells:
                        if cell.data_type == "f":  # metadata strings are never formulas
                            cell.data_type = "s"
                        if isinstance(cell.value, float):
                            cell.number_format = ("0.########E+00" if cell.value != 0 and
                                                  (abs(cell.value) < 1e-8 or abs(cell.value) >= 1e10)
                                                  else "0.00000000")
        verify_workbook(temporary, {"INDEX": index, "README": readme, **sheets})
        verify_report_snapshot(all_rows)
        # A failed Excel write must not truncate the last known-good paper
        # workbook.  Publish the fully closed temporary file in one rename.
        os.replace(temporary, out_path)
    finally:
        temporary.unlink(missing_ok=True)
    n = sum(len(s) for s in sheets.values())
    print(f"Wrote {out_path} ({n} rows across {len(sheets)} experiment sheets; atomic replace).")
    for name, df in sheets.items():
        print(f"  {name}: {len(df)} rows")


if __name__ == "__main__":
    main()
