#!/usr/bin/env python3
"""Post-inference transport report: value vs tonnage, and per-lane error.

Reads finished run directories (``<outputs>/{agg,full,sweep}/<run>/``) and
writes, without touching the training pipeline:

* ``value_vs_tonnage.xlsx`` -- every run's value and weight (tonnage) metrics
  side by side, per-arm seed summaries for Exp 1/2/3, and how each arm's rank
  changes between the value and tonnage targets.
* ``lane_errors.xlsx`` + ``lane_error_*.png`` -- Exp 2 per-lane error on the
  normalized training scale (MSE, the training objective), stacked by lane
  size, plus a real-unit WAPE companion.

Lane size is ranked by training-period tonnage (``--train-end``) so the size
split never uses test data.

    python scripts/post_inference_report.py \
        --outputs /path/to/production-outputs \
        --data-csv data/processed/hs2_monthly_aggregated.csv \
        --out /path/to/report
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("value", "weight")
METRICS = ("mse", "rmse", "mae", "r2", "mase", "rmsse", "smape")
LANE = ["State", "Commodity", "Import/Export"]
TIERS = ("Top 10% tonnage", "Middle 40%", "Bottom 50%")
TIER_COLORS = ("#1f4e79", "#6fa8dc", "#cfe2f3")


def run_rows(outputs):
    rows = []
    for exp, sub in (("Exp1", "agg"), ("Exp2", "full"), ("Exp3", "sweep")):
        for path in sorted((outputs / sub).glob("*/metrics.json")):
            payload = json.loads(path.read_text())
            metrics = payload.get("metrics", {})
            name = path.parent.name
            row = {
                "experiment": exp,
                "run": name,
                "model": payload.get("model") or name.split("_")[0],
                "variant": payload.get("variant", ""),
                "seed": payload.get("seed"),
            }
            grid = re.search(r"_w(\d+)_l(\d+)_seed", name)
            row["window"], row["lags"] = (int(grid[1]), int(grid[2])) if grid else (None, None)
            for target in TARGETS:
                for metric in METRICS:
                    row[f"{target}_{metric}"] = metrics.get(f"{target}_{metric}")
            row["combined_norm_mse"] = metrics.get("norm_mse_combined")
            row["combined_r2"] = metrics.get("r2")
            rows.append(row)
    return pd.DataFrame(rows)


def arm_summary(runs, experiment):
    keys = ["model", "variant"] + (["window", "lags"] if experiment == "Exp3" else [])
    data = runs[runs.experiment == experiment]
    cols = [f"{t}_{m}" for t in TARGETS for m in ("mse", "r2", "mase", "rmsse")] + ["combined_norm_mse"]
    grouped = data.groupby(keys, dropna=False)[cols]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_sd"))
    summary.insert(0, "seeds", data.groupby(keys, dropna=False).size())
    summary = summary.reset_index()
    summary["rank_value_mse"] = summary["value_mse_mean"].rank(method="min")
    summary["rank_weight_mse"] = summary["weight_mse_mean"].rank(method="min")
    summary["rank_change_value_to_tonnage"] = summary["rank_value_mse"] - summary["rank_weight_mse"]
    return summary.sort_values("weight_mse_mean")


def completed_predictions(root):
    """Prediction files of finished runs only.

    Inference writes predictions.csv before metrics.json (the completion
    marker), so a run interrupted in between must not be reported.
    """
    return sorted(p for p in root.glob("*/predictions.csv") if (p.parent / "metrics.json").is_file())


def training_rows(raw, train_start, train_end):
    return raw[(raw.Time >= pd.Timestamp(train_start)) & (raw.Time <= pd.Timestamp(train_end))]


def span_label(train_start, train_end):
    return f"{train_start[:4]}-{train_end[:4]}"


def lane_sizes(data_csv, train_start, train_end):
    raw = pd.read_csv(data_csv, parse_dates=["Time"])
    train = training_rows(raw, train_start, train_end)
    size = train.groupby(LANE, as_index=False)[["Weight", "Value"]].sum()
    size = size.rename(columns={"Weight": "train_tonnage_kg", "Value": "train_value_usd"})
    return size


def assign_tiers(frame):
    pct = frame["train_tonnage_kg"].rank(pct=True, method="first")
    frame["size_tier"] = np.select([pct > 0.9, pct > 0.5], TIERS[:2], TIERS[2])
    return frame


def is_real_unit(pred):
    """Upstream S4 evaluation writes real units into the unsuffixed columns too."""
    return all(np.allclose(pred[f"pred_{t}"], pred[f"pred_{t}_real"]) for t in TARGETS)


def lane_scalers(outputs):
    """Per-lane affine map real -> normalized, recovered from native runs.

    Every Exp 2 run shares the same train-fitted per-lane minmax scaling, so
    fitting (real, normalized) pairs from native prediction files recovers it
    exactly; hydra S4 files (real units only) are then put on the same scale.
    """
    pairs = []
    for path in completed_predictions(outputs / "full"):
        pred = pd.read_csv(path)
        if is_real_unit(pred):
            continue
        for kind in ("pred", "true"):
            for target in TARGETS:
                part = pred[LANE + [f"{kind}_{target}_real", f"{kind}_{target}"]].copy()
                part.columns = LANE + ["real", "norm"]
                part["target"] = target
                pairs.append(part)
        if len(pairs) >= 40:  # 10 native runs x (pred,true) x 2 targets
            break
    pairs = pd.concat(pairs, ignore_index=True)
    rows = []
    for (state, commodity, flow, target), grp in pairs.groupby(LANE + ["target"]):
        x, y = grp.real.to_numpy(), grp.norm.to_numpy()
        if np.ptp(x) == 0:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        resid = np.max(np.abs(slope * x + intercept - y))
        rows.append({"State": state, "Commodity": commodity, "Import/Export": flow,
                     "target": target, "slope": slope, "intercept": intercept, "max_resid": resid})
    scalers = pd.DataFrame(rows)
    worst = scalers.max_resid.max()
    if worst > 1e-6:
        raise ValueError(f"per-lane scaling is not a single affine map (max residual {worst:.2e})")
    return scalers


def to_normalized(pred, scalers):
    pred = pred.copy()
    for target in TARGETS:
        s = scalers[scalers.target == target][LANE + ["slope", "intercept"]]
        merged = pred[LANE].merge(s, on=LANE, how="left")
        if merged.slope.isna().any():
            raise ValueError(f"missing {target} scaler for {merged.slope.isna().sum()} rows")
        for kind in ("pred", "true"):
            pred[f"{kind}_{target}"] = merged.slope.to_numpy() * pred[f"{kind}_{target}_real"].to_numpy() \
                + merged.intercept.to_numpy()
    return pred


def lane_errors(outputs, sizes):
    per_run = []
    scalers = lane_scalers(outputs)
    for path in completed_predictions(outputs / "full"):
        meta = json.loads((path.parent / "metrics.json").read_text())
        pred = pd.read_csv(path)
        if is_real_unit(pred):
            pred = to_normalized(pred, scalers)
        for target in TARGETS:
            pred[f"se_{target}"] = (pred[f"pred_{target}"] - pred[f"true_{target}"]) ** 2
            pred[f"ae_{target}_real"] = (pred[f"pred_{target}_real"] - pred[f"true_{target}_real"]).abs()
        lane = pred.groupby(LANE).agg(
            mse_value=("se_value", "mean"), mse_weight=("se_weight", "mean"),
            abs_err_value_real=("ae_value_real", "sum"), abs_err_weight_real=("ae_weight_real", "sum"),
            true_value_real=("true_value_real", "sum"), true_weight_real=("true_weight_real", "sum"),
        ).reset_index()
        lane["mse_combined"] = (lane.mse_value + lane.mse_weight) / 2
        for target in TARGETS:
            total = lane[f"true_{target}_real"]
            lane[f"wape_{target}_pct"] = np.where(total > 0, 100 * lane[f"abs_err_{target}_real"] / total.where(total > 0), np.nan)
        lane.insert(0, "seed", meta.get("seed"))
        lane.insert(0, "variant", meta.get("variant", ""))
        lane.insert(0, "model", meta.get("model") or path.parent.name.split("_")[0])
        lane.insert(0, "run", path.parent.name)
        per_run.append(lane)
    lanes = pd.concat(per_run, ignore_index=True).merge(sizes, on=LANE, how="left")
    return assign_tiers_by_lane(lanes, sizes)


def assign_tiers_by_lane(lanes, sizes):
    cohort = lanes[LANE].drop_duplicates().merge(sizes, on=LANE, how="left")
    cohort = assign_tiers(cohort)
    return lanes.merge(cohort[LANE + ["size_tier"]], on=LANE, how="left")


def seed_mean(lanes):
    cols = ["mse_value", "mse_weight", "mse_combined", "wape_value_pct", "wape_weight_pct"]
    return (lanes.groupby(["model", "variant"] + LANE + ["size_tier", "train_tonnage_kg"], dropna=False)[cols]
            .mean().reset_index())


def tier_table(arm_lanes):
    rows = []
    for (model, variant), group in arm_lanes.groupby(["model", "variant"]):
        for tier in TIERS + ("All lanes",):
            part = group if tier == "All lanes" else group[group.size_tier == tier]
            rows.append({
                "model": model, "variant": variant, "size_tier": tier, "lanes": len(part),
                "mse_weight_mean": part.mse_weight.mean(), "mse_weight_median": part.mse_weight.median(),
                "mse_value_mean": part.mse_value.mean(), "mse_value_median": part.mse_value.median(),
                "wape_weight_median_pct": part.wape_weight_pct.median(),
                "wape_value_median_pct": part.wape_value_pct.median(),
                "tonnage_weighted_mse_weight": np.average(part.mse_weight, weights=part.train_tonnage_kg)
                if len(part) and part.train_tonnage_kg.sum() > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def plot_arm(arm, title, path, span):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    positive = arm[["mse_weight", "mse_value"]].to_numpy()
    positive = positive[positive > 0]
    lo = np.floor(np.log10(positive.min())) if positive.size else -6
    hi = np.ceil(np.log10(positive.max())) if positive.size else 0
    mse_bins = np.logspace(lo, hi, 31)
    wape_bins = np.append(np.arange(0, 210, 10), 1e9)
    panels = (("mse_weight", "Tonnage: per-lane MSE (normalized, training loss)", mse_bins, True),
              ("mse_value", "Value: per-lane MSE (normalized, training loss)", mse_bins, True),
              ("wape_weight_pct", "Tonnage: per-lane WAPE % (real units)", wape_bins, False),
              ("wape_value_pct", "Value: per-lane WAPE % (real units)", wape_bins, False))
    for ax, (col, label, bins, log) in zip(axes.flat, panels):
        data = [arm.loc[arm.size_tier == tier, col].dropna().clip(upper=bins[-2] + 1 if not log else None)
                for tier in TIERS]
        ax.hist(data, bins=bins if log else np.append(np.arange(0, 210, 10), 210),
                stacked=True, color=TIER_COLORS, label=TIERS, edgecolor="white", linewidth=0.4)
        if log:
            ax.set_xscale("log")
        else:
            ax.set_xticks(range(0, 211, 20))
            ax.set_xticklabels([str(t) if t < 200 else ">200" for t in range(0, 211, 20)])
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Number of lanes")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + f"  ({len(arm)} lanes, mean of seeds; size = {span} tonnage)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


BASELINE_MODELS = {"seasonal_naive", "random_walk", "moving_average"}


def family(experiment, model, variant):
    if model in BASELINE_MODELS:
        return f"Baseline ({experiment})"
    if experiment == "Exp1":
        return "Aggregated (Exp1, one total series)"
    if model == "s4nd":
        return "ND state-space (S4ND)"
    if model == "mamba_nd":
        # Its variants are named cross_attention_2d/3d/4d but are per-axis Mamba scans.
        return "ND per-axis scan (Mamba-ND)"
    if variant.startswith("cross_attention") or variant == "film_attention":
        return "Fusion (cross-attention / FiLM)"
    return "All-in-one per-lane (embeddings / one-hot)"


def aggregate_vs_lanes(outputs, data_csv, train_start, train_end, out_dir):
    """Compare the aggregated model (Exp1) with lane models summed bottom-up (Exp2).

    Common target: the monthly total of the Exp2 cohort (918 lanes). Exp2
    forecasts are summed over lanes; Exp1 forecasts the all-lane total, so it
    is scaled by the cohort's training-period share (no test data used).
    """
    agg_files = completed_predictions(outputs / "agg")
    lane_files = completed_predictions(outputs / "full")
    if not agg_files or not lane_files:
        return None
    raw = pd.read_csv(data_csv, parse_dates=["Time"])
    cohort = pd.read_csv(lane_files[0])[LANE].drop_duplicates()
    raw["in_cohort"] = raw.set_index(LANE).index.isin(cohort.set_index(LANE).index)
    train = training_rows(raw, train_start, train_end)
    share = {t: train.loc[train.in_cohort, t.capitalize()].sum() / train[t.capitalize()].sum() for t in TARGETS}
    test = raw[raw.Time >= pd.read_csv(lane_files[0]).Time.min()]
    truth = test[test.in_cohort].groupby("Time")[["Value", "Weight"]].sum()
    truth.index = truth.index.strftime("%Y-%m-%d")

    # Exp1's own normalized scale for the total (train-fitted minmax), recovered exactly.
    ref = pd.read_csv(agg_files[0])
    scale = {t: np.polyfit(np.r_[ref[f"pred_{t}_real"], ref[f"true_{t}_real"]],
                           np.r_[ref[f"pred_{t}"], ref[f"true_{t}"]], 1) for t in TARGETS}

    rows = []
    for experiment, files in (("Exp1", agg_files), ("Exp2", lane_files)):
        for path in files:
            meta = json.loads((path.parent / "metrics.json").read_text())
            pred = pd.read_csv(path)
            if experiment == "Exp1":
                total = pred.set_index("Time")[[f"pred_{t}_real" for t in TARGETS]]
                for t in TARGETS:
                    total[f"pred_{t}_real"] *= share[t]
            else:
                total = pred.groupby("Time")[[f"pred_{t}_real" for t in TARGETS]].sum()
            model, variant = meta.get("model") or path.parent.name.split("_")[0], meta.get("variant", "")
            row = {"experiment": experiment, "family": family(experiment, model, variant),
                   "model": model, "variant": variant, "seed": meta.get("seed"), "run": path.parent.name}
            for t in TARGETS:
                y = truth[t.capitalize()].reindex(total.index).to_numpy(float)
                p = total[f"pred_{t}_real"].to_numpy(float)
                err = p - y
                slope = scale[t][0]
                row[f"{t}_mae"] = np.abs(err).mean()
                row[f"{t}_mse"] = (err ** 2).mean()
                row[f"{t}_mae_norm"] = np.abs(err * slope).mean()
                row[f"{t}_mse_norm"] = ((err * slope) ** 2).mean()
                row[f"{t}_wape_pct"] = 100 * np.abs(err).sum() / np.abs(y).sum()
                row[f"{t}_bias_pct"] = 100 * err.sum() / y.sum()
            rows.append(row)
    runs = pd.DataFrame(rows)
    cols = [c for c in runs.columns if c.startswith(TARGETS)]
    grouped = runs.groupby(["family", "experiment", "model", "variant"], dropna=False)
    arms = grouped[cols].mean().add_suffix("_mean").join(grouped[cols].std().add_suffix("_sd"))
    arms.insert(0, "seeds", grouped.size())
    arms = arms.reset_index().sort_values("weight_mse_norm_mean")
    fams = runs.groupby("family")[cols].agg(["mean", "median", "min"])
    fams.columns = [f"{a}_{b}" for a, b in fams.columns]
    fams.insert(0, "runs", runs.groupby("family").size())
    fams.insert(1, "arms", runs.groupby("family")[["model", "variant"]].apply(lambda g: len(g.drop_duplicates())))
    best = arms.loc[arms.groupby("family").weight_mse_norm_mean.idxmin()].sort_values("weight_mse_norm_mean")

    notes = pd.DataFrame({"note": [
        "Common target: monthly total of the 918 Exp2 lanes, Jan-May 2025 (both models judged on the same numbers).",
        "Exp2 (lane / fusion / ND models): lane forecasts summed per month (bottom-up).",
        f"Exp1 (aggregated): all-lane total forecast x cohort training share "
        f"(value {share['value']:.4f}, weight {share['weight']:.4f}; {span_label(train_start, train_end)}, no test data).",
        "*_mae / *_mse: real units (US$ for value, kg for weight). *_norm: on Exp1's own normalized "
        "(minmax, training-loss) scale for the total.",
        "*_wape_pct: sum|error| / sum(actual); *_bias_pct > 0 means over-forecast.",
        "Only 5 test months: treat differences as descriptive until tested (see docs/STATISTICAL_PROTOCOL.md).",
    ]})
    path = out_dir / "aggregate_vs_lanes.xlsx"
    with pd.ExcelWriter(path) as xl:
        notes.to_excel(xl, sheet_name="Notes", index=False)
        best.to_excel(xl, sheet_name="Best_arm_per_family", index=False)
        fams.reset_index().sort_values("weight_mse_norm_mean").to_excel(xl, sheet_name="By_family", index=False)
        arms.to_excel(xl, sheet_name="By_arm", index=False)
        runs.to_excel(xl, sheet_name="All_runs", index=False)
        truth.to_excel(xl, sheet_name="Target_918_lane_total")
    plot_aggregate_vs_lanes(runs, out_dir / "aggregate_vs_lanes.png")
    return path


def plot_aggregate_vs_lanes(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = runs.groupby("family").weight_mse_norm.median().sort_values().index
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (col, label) in zip(axes.flat, (("weight_mae_norm", "Tonnage MAE (normalized)"),
                                           ("weight_mse_norm", "Tonnage MSE (normalized)"),
                                           ("value_mae_norm", "Value MAE (normalized)"),
                                           ("value_mse_norm", "Value MSE (normalized)"))):
        data = [runs.loc[runs.family == f, col].to_numpy() for f in order]
        ax.boxplot(data, vert=False, showfliers=True)
        ax.set_yticks(range(1, len(order) + 1))
        ax.set_yticklabels([f.replace(" (", "\n(") for f in order], fontsize=7)
        ax.set_title(label + " of the 918-lane total, Jan-May 2025", fontsize=9)
        ax.invert_yaxis()
    fig.suptitle("Aggregated model vs lane / fusion / ND models summed bottom-up (every run; lower is better)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--train-start", default="2009-01-01")
    parser.add_argument("--train-end", default="2023-12-01")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = run_rows(args.outputs)
    notes = pd.DataFrame({"note": [
        "value_* = trade value (US$); weight_* = shipping weight (kg), i.e. tonnage.",
        "MSE/RMSE/MAE/R2 are on the normalized (minmax) training scale; MSE is the training loss.",
        "MASE/RMSSE compare against a naive seasonal forecast: below 1 beats it.",
        "rank_change_value_to_tonnage > 0: the arm ranks better on tonnage than on value.",
        "Target is a mixed vessel/containerized value-and-weight measure (docs/SOURCE_FIELD_AUDIT.md).",
        "2025 test results are retrospective, not a real-time forecast.",
        f"Runs found: Exp1 {sum(runs.experiment == 'Exp1')}, Exp2 {sum(runs.experiment == 'Exp2')}, "
        f"Exp3 {sum(runs.experiment == 'Exp3')}.",
    ]})
    with pd.ExcelWriter(args.out / "value_vs_tonnage.xlsx") as xl:
        notes.to_excel(xl, sheet_name="Notes", index=False)
        for exp in ("Exp1", "Exp2", "Exp3"):
            if (runs.experiment == exp).any():
                arm_summary(runs, exp).to_excel(xl, sheet_name=f"{exp}_value_vs_tonnage", index=False)
        runs.to_excel(xl, sheet_name="All_runs", index=False)

    comparison = aggregate_vs_lanes(args.outputs, args.data_csv, args.train_start, args.train_end, args.out)
    if comparison:
        print(f"Wrote {comparison}")
    if not completed_predictions(args.outputs / "full"):
        print("No Exp 2 predictions found; wrote value_vs_tonnage.xlsx only.")
        return
    sizes = lane_sizes(args.data_csv, args.train_start, args.train_end)
    span = span_label(args.train_start, args.train_end)
    lanes = lane_errors(args.outputs, sizes)
    arms = seed_mean(lanes)
    tiers = tier_table(arms)
    with pd.ExcelWriter(args.out / "lane_errors.xlsx") as xl:
        pd.DataFrame({"note": [
            "Per-lane errors over the 5 test months (Jan-May 2025), Exp 2 (918 lanes).",
            "mse_* uses the normalized training scale (same scale as the training MSE loss).",
            "wape_*_pct = sum|error| / sum(actual) in real units; blank when the lane had no test flow.",
            f"Size tiers rank lanes by {span} training tonnage (no test data): top 10%, next 40%, bottom 50%.",
        ]}).to_excel(xl, sheet_name="Notes", index=False)
        tiers.to_excel(xl, sheet_name="By_size_tier", index=False)
        arms.to_excel(xl, sheet_name="Per_lane_seed_mean", index=False)
        lanes.to_excel(xl, sheet_name="Per_lane_per_run", index=False)
    figs = args.out / "lane_error_figures"
    figs.mkdir(exist_ok=True)
    for (model, variant), arm in arms.groupby(["model", "variant"]):
        plot_arm(arm, f"{model} / {variant}", figs / f"lane_error_{model}_{variant}.png", span)
    print(f"Wrote {args.out}/value_vs_tonnage.xlsx, lane_errors.xlsx and {len(list(figs.glob('*.png')))} figures.")


if __name__ == "__main__":
    main()
