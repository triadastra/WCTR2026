"""Source-validated Exp 2 reporting.

Primary loss is lane-weighted MSE: squared errors are averaged over valid
series-time observations and the two targets, without equalizing states.
Baseline and axial comparisons use that same mean with state-clustered
uncertainty. Equal-state Friedman/Nemenyi ranks are a separately labeled
sensitivity analysis; neither its non-rejection nor a CD group implies
model equivalence. Three-seed intervals describe training variation.
"""
import math
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ext.eval.resampling import (friedman_nemenyi, paired_t, cluster_robust_paired_t,
                                 wild_cluster_bootstrap, cluster_bootstrap)  # noqa: E402
from src.utils.exp2_artifacts import validated_exp2_artifacts  # noqa: E402

# parents[1]: this script lives in util/, the repo root is one level up.
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "full"
OUTDIR = ROOT / "graphs" / "_analysis"
SEEDS = (26, 105, 115)
KEY = ["State", "Commodity", "Import/Export", "Time"]
T_975_2 = 4.302653   # Student-t 0.975 quantile, 2 dof (n=3 -> n-1=2)

FAMILY = {
    "mamba": "SSM", "mamba_nd": "SSM", "s4": "SSM", "s4nd": "SSM",
    "lstm": "RNN", "gru": "RNN", "transformer": "Attention", "gpt": "Attention",
    "xgboost": "Tree", "lightgbm": "Tree", "random_forest": "Tree",
    "moving_average": "Baseline", "random_walk": "Baseline", "seasonal_naive": "Baseline",
}
FAM_COLOR = {"SSM": "#34a853", "RNN": "#4285f4", "Attention": "#ea4335",
             "Tree": "#fbbc04", "Baseline": "#9aa0a6"}


def point_loss(df, metric="mse"):
    error = df[["pred_value", "pred_weight"]].to_numpy() - df[["true_value", "true_weight"]].to_numpy()
    if metric not in ("mse", "mae"):
        raise ValueError("metric must be mse or mae")
    return pd.Series((error ** 2 if metric == "mse" else np.abs(error)).mean(axis=1), index=df.index)


def load_model_losses(metric="mse"):
    """Read only the complete canonical, source-validated arm/seed roster."""
    groups = {}
    for run, path, payload in validated_exp2_artifacts(ROOT, RUNS):
        arm = f"{run.model}_{run.variant}"
        df = pd.read_csv(path.parent / payload["predictions_file"])
        loss = df.assign(_loss=point_loss(df, metric)).set_index(KEY)["_loss"].sort_index()
        if not loss.index.is_unique or not np.isfinite(loss).all():
            raise RuntimeError(f"{arm}: duplicate identities or non-finite losses")
        seeds = groups.setdefault(arm, {})
        if run.seed in seeds:
            raise RuntimeError(f"{arm}: duplicate seed {run.seed}")
        seeds[run.seed] = loss
    out, reference = {}, None
    for arm, seeds in groups.items():
        avg, seed_means, reference = average_seed_losses(arm, seeds, reference)
        out[arm] = (avg, seed_means)
    return out


def average_seed_losses(arm, losses_by_seed, reference_index=None):
    """Average one arm's per-point losses over its seeds, failing closed.

    ``losses_by_seed`` maps seed -> per-point loss Series indexed by KEY.
    Refuses an incomplete roster (a two-seed arm averaged next to three-seed
    arms is a different, noisier estimate under the same column name, and a
    missing seed usually means a run that died) and refuses any seed whose
    (lane, month) population differs from ``reference_index`` -- intersecting
    silently would let one run's missing targets shrink the population every
    other arm is compared on. Returns (mean_loss, per_seed_means, index).
    """
    if set(losses_by_seed) != set(SEEDS):
        raise RuntimeError(
            f"{arm}: incomplete paired seed roster {sorted(losses_by_seed)}; "
            f"expected {list(SEEDS)}"
        )
    series, seed_means = [], []
    for s in SEEDS:
        loss = losses_by_seed[s].sort_index()
        if not loss.index.is_unique or not np.isfinite(loss).all():
            raise RuntimeError(f"{arm} seed {s}: duplicate identities or non-finite losses")
        if reference_index is not None and not loss.index.equals(reference_index):
            raise RuntimeError(f"{arm} seed {s}: prediction population differs from the roster")
        reference_index = loss.index
        series.append(loss)
        seed_means.append(float(loss.mean()))
    return sum(series) / len(series), seed_means, reference_index


def per_combo(loss):
    """Descriptive lane means, averaging repeated months first."""
    return loss.groupby(level=[0, 1, 2]).mean()


def per_state(loss):
    """Equal-state sensitivity: average months within lanes, then lanes within states."""
    lane = per_combo(loss)
    return lane.groupby(level=0).mean()


def aligned_difference(left, right):
    """Reject population loss instead of intersecting away missing blocks."""
    left, right = left.sort_index(), right.sort_index()
    if not left.index.is_unique or not left.index.equals(right.index):
        raise RuntimeError("Paired populations differ")
    return (left - right).to_numpy()


def paired_comparison(left, right):
    """Compare left minus right on every paired valid observation.

    State clustering changes uncertainty, not the point-estimate weights.
    Using observations directly also preserves the declared objective when
    series have unequal observation counts. Equal-state means are secondary.
    """
    left, right = left.sort_index(), right.sort_index()
    d = aligned_difference(left, right)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError("Paired comparison requires finite observations")
    clusters = left.index.get_level_values("State").to_numpy()
    t, p_t, mean, count = cluster_robust_paired_t(d, clusters)
    p_w, _, _, _, reps = wild_cluster_bootstrap(d, clusters)
    p_b, _, lo, hi = cluster_bootstrap(d, clusters=clusters, n_boot=10000, seed=26)
    state_diff = aligned_difference(per_state(left), per_state(right))
    _, p_state, mean_state = paired_t(state_diff)
    return dict(mean_delta_mse=mean, mean_lane_delta=mean,
                mean_state_delta=mean_state, p_state_sensitivity=p_state,
                t=t, p_cluster_t=p_t, p_wild_cluster=p_w,
                p_bootstrap=p_b, boot_ci95=f"[{lo:+.6f}, {hi:+.6f}]",
                degenerate=not np.isfinite(p_w), n_blocks=count,
                n_clusters=count, n_observations=len(d),
                n_lanes=len(left.index.droplevel("Time").unique()),
                inference_unit="series_time", cluster_unit="state",
                primary_estimand="lane_weighted", inference_status="exploratory",
                n_wild_replications=reps)


def state_loss_matrix(data, arms=None):
    """One equally weighted state block per arm; lanes/months are nested."""
    arms = sorted(data) if arms is None else list(arms)
    blocks = [per_state(data[arm][0]).sort_index() for arm in arms]
    if not blocks:
        raise RuntimeError("No validated arms")
    for block in blocks[1:]:
        aligned_difference(blocks[0], block)
    return np.column_stack([block.to_numpy() for block in blocks])


def significance_table(data):
    means = {arm: float(value[0].mean()) for arm, value in data.items()}
    order = sorted(means, key=means.get)
    result = friedman_nemenyi(state_loss_matrix(data, order))
    ranks = dict(zip(order, result["mean_ranks"]))
    reference = min(order, key=ranks.get)
    rows = []
    for arm in order:
        seeds = data[arm][1]
        ci = T_975_2 * st.stdev(seeds) / math.sqrt(len(seeds))
        rows.append(dict(
            run=arm, model=next((m for m in sorted(FAMILY, key=len, reverse=True)
                                if arm.startswith(m + "_")), arm),
            mean_norm_mse=means[arm], ci95_halfwidth=ci, n_seeds=len(seeds),
            mean_rank=ranks[arm], rank_reference=reference,
            delta_vs_rank_reference=means[arm] - means[reference],
            nemenyi_cd=float(result["cd"]),
            within_cd_of_reference=abs(ranks[arm] - ranks[reference]) <= result["cd"],
            inference_unit="state", n_blocks=result["n"], alpha=result["alpha"],
            friedman_p=result["p"], inference_status="sensitivity",
            primary_estimand="lane_weighted", rank_estimand="equal_state",
            asymptotic_reliability_condition_met=result["asymptotic_reliability_condition_met"],
        ))
    return pd.DataFrame(rows)


def main():
    data = load_model_losses()
    table = significance_table(data)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTDIR / "significance.csv", index=False)
    print(table.to_string(index=False))
    fig, ax = plt.subplots(figsize=(11, max(5, len(table) * .25)))
    y = np.arange(len(table))[::-1]
    ax.errorbar(table.mean_norm_mse, y, xerr=table.ci95_halfwidth,
                fmt="o", capsize=3, markersize=4)
    ax.set_yticks(y, table.run, fontsize=7)
    ax.set_xlabel("Pooled normalized MSE; whiskers: descriptive 3-seed Student-t 95% interval")
    ax.set_title("Equal-state rank sensitivity (Friedman–Nemenyi)\n"
                 "Eight states share national shocks; non-rejection does not establish equivalence")
    ax.grid(axis="x", linestyle=":", alpha=.4)
    fig.tight_layout()
    fig.savefig(OUTDIR / "significance.png", dpi=160)
    plt.close(fig)
    # Standalone outputs avoid attaching fresh statistics to an unchecked workbook.


if __name__ == "__main__":
    main()
