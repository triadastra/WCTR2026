#!/usr/bin/env python3
"""Diebold-Mariano test harness for the WCTR-2026 per-series benchmark.

READ-ONLY consumer of outputs/full/<run>/predictions.csv. Computes DM
statistics on the combined (value+weight) one-step squared-error loss, which is
exactly the headline metric:  L = 0.5*((pv-tv)^2 + (pw-tw)^2),  mean(L)=norm_mse_combined.

Protocol (corrected):
  * Comparator is PRE-REGISTERED, not data-driven: every arm is tested against
    the ``moving_average`` naive baseline, the one naive baseline that
    discriminates on this panel (random walk and seasonal naive lose to every
    arm; see FINDINGS.md). An earlier version of this script instead picked a
    per-family "champion" by minimum TEST loss and then ran DM against that
    champion on the same test losses that selected it -- a selection-on-test
    bias -- and silently kept one arbitrary (lexicographically first) seed per
    arm. Both defects are removed: nothing here is chosen using test data, and
    every seed of an arm contributes.
  * Seeds are averaged: an arm's loss at each (series, month) observation is
    the mean over all of that arm's seed runs, matching how the headline table
    aggregates seeds.
  * The primary p-values are series-clustered (each of the kept series is one
    unit; months within a series are correlated and must not be pooled) and
    Holm-corrected across the family of arm-vs-baseline comparisons. Pooled
    z/p are reported for reference only.
  * The all-pairs clustered-p matrix is DESCRIPTIVE (appendix): it involves no
    data-driven selection, but it is not multiplicity-corrected -- simultaneous
    leaderboard inference lives in util/make_significance.py
    (Friedman + Nemenyi).

Two DM variants per comparison:
  * pooled    : every aligned (series, month) observation is a unit
  * clustered : series-mean loss differentials are the units -> respects the
                panel structure, avoids over-counting correlated within-series
                months. This is the PRIMARY result.

Purely additive: writes only under ext/outputs/dm/. Modifies nothing.
"""
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ext.eval.resampling import holm  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    corrected_per_combo_artifact,
    corrected_prediction_artifact,
)

# Default to THIS repository (ext/eval/ -> repo root), not a machine-specific
# path. The old ~/full default meant a fresh clone silently found zero runs and
# exited on the "comparator not found" branch instead of doing anything.
REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[2]))
FULL_DIR = os.path.join(REPO, "outputs", "full")
# S4/S4ND predictions are dumped here by ext/eval/s4_dump_preds.py (keeps
# outputs/full pristine -- s4_eval only saves metrics.json, not per-series CSV).
S4_PRED_DIR = os.path.join(REPO, "ext", "outputs", "s4_preds")
OUT_DIR = os.path.join(REPO, "ext", "outputs", "dm")

# Pre-registered comparator. Fixed a priori; never derived from the losses.
#
# The value must be the ARM KEY, which is the run directory minus its _seed<N>
# suffix -- "moving_average_none", not "moving_average". With the bare name this
# lookup never matched anything, so the script always exited on the
# "comparator not found" branch and could not run at all.
BASELINE_ARM = "moving_average_none"


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


# Model-name prefixes, matched LONGEST FIRST. Splitting on the first "_" and
# looking the token up is what this used to do, and it silently mislabelled two
# families: "random_forest_onehot" -> "random" -> RandomWalk (a naive BASELINE),
# and "mamba_nd_cross_attention_2d" -> "mamba" -> Mamba. Both appeared in the
# emitted CSV under another model's name.
_FAMILY_BY_PREFIX = {
    "lstm": "LSTM", "gru": "GRU", "transformer": "Transformer", "gpt": "GPT",
    "dss": "DSS", "mamba_nd": "Mamba-ND", "mamba": "Mamba",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "RandomForest",
    "moving_average": "MovingAvg", "random_walk": "RandomWalk",
    "seasonal_naive": "SeasonalNaive",
}


def family_of(run):
    """Collapse a run-dir name (minus _seed<N>) into a model family label."""
    r = run
    if r.startswith("s4_s4nd_2d"): return "S4ND-2D"
    if r.startswith("s4_s4nd_3d"): return "S4ND-3D"
    if r.startswith("s4_s4nd_4d"): return "S4ND-4D"
    if r.startswith("s4_s4_"):     return "S4"
    for prefix in sorted(_FAMILY_BY_PREFIX, key=len, reverse=True):
        if r == prefix or r.startswith(prefix + "_"):
            return _FAMILY_BY_PREFIX[prefix]
    return r.split("_")[0]


def normalized_predictions_path(run_dir):
    """Path to the run's NORMALIZED prediction table, or None if it has none.

    This is not a formality. The S4 family writes TWO tables into the same run
    directory: ``predictions.csv`` in real dollars/kg, and ``predictions_norm.csv``
    on the per-combo [0,1] scale everything else reports. Reading the former for
    an s4/s4nd arm puts its loss ~10^14x above the rest of the panel, and every
    ratio and z-statistic computed from it is nonsense. The run's metrics.json
    names the correct table in ``predictions_file``; fall back to the normalized
    name, then to predictions.csv only for runs that have no normalized twin
    (the github + tree arms, whose predictions.csv IS the normalized table when
    they were evaluated with --normalized).
    """
    mj = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mj):
        try:
            with open(mj) as f:
                named = json.load(f).get("predictions_file")
        except (OSError, ValueError):
            named = None
        if named and os.path.exists(os.path.join(run_dir, named)):
            return os.path.join(run_dir, named)
    for name in ("predictions_norm.csv", "predictions.csv"):
        cand = os.path.join(run_dir, name)
        if os.path.exists(cand):
            return cand
    return None


# Normalized targets are min-max scaled on train, so val/test can overshoot 1
# but never approach 100. Anything above this is real units and must not be
# pooled with the rest of the panel. Mirrors the guard in
# scripts/build_paper_results.py::_ensemble.
_REAL_UNIT_TRIPWIRE = 100.0


def load_run(run_dir, prediction_path=None):
    """Return dict: key=(State,Commodity,Flow,Time) -> combined loss L."""
    path = prediction_path or normalized_predictions_path(run_dir)
    if path is None:
        raise FileNotFoundError(f"{run_dir}: no prediction table found")
    out = {}
    max_true = 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                pv = float(row["pred_value"]);  tv = float(row["true_value"])
                pw = float(row["pred_weight"]); tw = float(row["true_weight"])
            except (KeyError, ValueError):
                continue
            key = (row["State"], row["Commodity"], row["Import/Export"], row["Time"])
            out[key] = 0.5 * ((pv - tv) ** 2 + (pw - tw) ** 2)
            max_true = max(max_true, abs(tv), abs(tw))
    if max_true > _REAL_UNIT_TRIPWIRE:
        raise SystemExit(
            f"{path} is in REAL units (max |true| = {max_true:.3g}), not the "
            f"per-combo normalized [0,1] scale this test pools over. Emit the "
            f"normalized table (scripts/s4_eval.py / ext/eval/s4_dump_preds.py) "
            f"and record it as `predictions_file` in metrics.json."
        )
    return out


def load_arms():
    """Load every run dir, grouped by arm (name minus _seed<N>), ALL seeds.

    Returns {arm: {key: seed-mean loss}}. An arm's loss profile is the mean
    over its seed runs at each key, restricted to keys present in every seed
    (identical by construction under one protocol; the intersection is a
    guard against mixed-vintage artifacts).
    """
    seed_runs = defaultdict(list)
    search_dirs = sorted(glob.glob(os.path.join(FULL_DIR, "*_seed*")))
    search_dirs += sorted(glob.glob(os.path.join(S4_PRED_DIR, "*_seed*")))
    seen_dirnames = set()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        if base in seen_dirnames:
            continue  # same run dumped in both trees: outputs/full wins
        # PROVENANCE GATE. Every other consumer of outputs/full refuses runs
        # that were evaluated on the pre-correction, full-timeline cohort
        # (903 lanes / 4,515 targets, selected with val/test availability in
        # view); this script did not, and would happily have produced DM
        # statistics on the survivorship-biased population. It never surfaced
        # only because BASELINE_ARM did not match its own directory naming, so
        # the run aborted before printing anything -- one bug hiding another.
        # metrics.json is REQUIRED, not optional. Making the check conditional
        # left a hole exactly where it mattered: ext/eval/s4_dump_preds.py
        # writes only predictions.csv into ext/outputs/s4_preds/<run>/, so a
        # supplemental dump with no matching run in outputs/full would sail
        # past this gate unverified. A legacy 903-lane dump entering here also
        # shrinks the all-arm key intersection below to the survivorship-biased
        # cohort -- quarantining that population is the whole point of the gate.
        # A directory that has predictions but cannot prove its cohort is
        # refused, not silently skipped: silently dropping an arm is the same
        # class of defect as filtering one out by its test loss.
        mj = os.path.join(d, "metrics.json")
        if not os.path.exists(mj):
            raise SystemExit(
                f"Refusing DM analysis: {base} has predictions but no "
                f"metrics.json, so its cohort cannot be verified. "
                f"scripts/s4_eval.py now writes predictions_norm.csv straight "
                f"into outputs/full with full provenance, which makes the "
                f"ext/outputs/s4_preds dumps redundant -- delete the "
                f"unprovenanced copy rather than scoring it."
            )
        with open(mj) as f:
            payload = json.load(f)
        current, reason = corrected_per_combo_artifact(payload)
        if not current:
            raise SystemExit(
                f"Refusing DM analysis of stale run {base}: {reason}. "
                f"Rerun it under the corrected train-only cohort "
                f"(see RERUN.md), or point REPO at a corrected runs tree."
            )
        current, reason = corrected_prediction_artifact(payload, Path(mj))
        if not current:
            raise SystemExit(
                f"Refusing DM analysis of incomplete run {base}: {reason}."
            )
        prediction_path = os.path.join(d, payload["predictions_file"])

        seen_dirnames.add(base)
        arm = base.rsplit("_seed", 1)[0]
        seed_runs[arm].append(load_run(d, prediction_path))

    arms, n_seeds = {}, {}
    for arm, runs in seed_runs.items():
        common = set(runs[0])
        for r in runs[1:]:
            common &= set(r)
        if not common:
            print(f"[warn] {arm}: no common keys across its {len(runs)} seed runs; skipped")
            continue
        if len(runs) > 1 and any(len(r) != len(common) for r in runs):
            print(f"[warn] {arm}: seed runs disagree on keys; using the "
                  f"{len(common)}-key intersection")
        arms[arm] = {k: float(np.mean([r[k] for r in runs])) for k in common}
        n_seeds[arm] = len(runs)
    return arms, n_seeds


def dm_clustered(losses_a, losses_b, keys):
    """Series-clustered DM. Each (State,Commodity,Flow) series is one unit."""
    by_series = defaultdict(list)
    for k in keys:
        by_series[(k[0], k[1], k[2])].append(losses_a[k] - losses_b[k])
    series_means = np.array([np.mean(v) for v in by_series.values()])
    n = len(series_means)
    mean_d = series_means.mean()
    sd = series_means.std(ddof=1)
    se = sd / math.sqrt(n) if sd > 0 else float("nan")
    z = mean_d / se if se and not math.isnan(se) else float("nan")
    return mean_d, z, two_sided_p(z) if not math.isnan(z) else float("nan"), n


def dm_pooled(losses_a, losses_b, keys):
    d = np.array([losses_a[k] - losses_b[k] for k in keys])
    n = len(d)
    mean_d = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    z = mean_d / se if se > 0 else float("nan")
    return mean_d, z, two_sided_p(z) if not math.isnan(z) else float("nan"), n


def main():
    print("This historical DM report is retired. Use util/make_vs_baseline.py "
          "for source-validated lane-weighted comparisons with state-clustered uncertainty.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
