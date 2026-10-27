"""Seasonal scale metrics shared by all benchmark runtimes."""
import numpy as np
import pandas as pd


def add_mase_rmsse(metrics, pipeline, pred_norm, targ_norm, keys):
    """Scale-free MASE / RMSSE, M4/M5 convention: scale EACH series by its own
    in-sample seasonal-naive (m=12) error, then AVERAGE across series (not a
    pooled ratio). Computed on the normalized train grid (MASE/RMSSE are
    scale-invariant). Missing history/scales are hard errors: silently emitting
    a completed artifact without the advertised scale-free metrics makes model
    rows incomparable.

    Shared by scripts/inference.py and scripts/run_baseline.py — previously it
    lived only here, so baseline rows carried NaN MASE/RMSSE in the workbook.
    Mutates ``metrics`` in place.
    """
    import collections

    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    targ_norm = np.asarray(targ_norm, dtype=np.float64)
    if pred_norm.shape != targ_norm.shape or pred_norm.ndim != 2 or pred_norm.shape[1] != 2:
        raise ValueError("MASE/RMSSE expects matching (n_samples, 2) arrays")
    if len(keys) != len(pred_norm):
        raise ValueError("MASE/RMSSE key count does not match prediction count")

    # MASE/RMSSE denominators use the entire in-sample history.  The
    # model-ready grid has already dropped lag burn-in months, so prefer the
    # retained pre-burn-in normalized grid and only fall back for legacy test
    # doubles that explicitly provide df_full_grid.
    g = getattr(pipeline, "df_metric_grid", None)
    if g is None:
        g = getattr(pipeline, "df_full_grid", None)
    if g is None:
        raise ValueError("pipeline has no grid from which to compute seasonal scales")
    tr = g[g["Time"] <= pd.Timestamp(pipeline.config.train_end)]
    scales = {}
    for (state, commodity, flow), group in tr.groupby(
        ["State", "Commodity", "Import/Export"], sort=False
    ):
        group = group.sort_values("Time")
        value = group["Value"].to_numpy(dtype=np.float64)
        weight = group["Weight"].to_numpy(dtype=np.float64)
        if len(value) <= 12:
            continue
        value_diff = value[12:] - value[:-12]
        weight_diff = weight[12:] - weight[:-12]
        scales[(str(state), str(commodity), str(flow))] = (
            float(np.mean(np.abs(value_diff))),
            float(np.mean(np.abs(weight_diff))),
            float(np.mean(value_diff ** 2)),
            float(np.mean(weight_diff ** 2)),
        )

    canonical_keys = [tuple(map(str, key)) for key in keys]
    missing = sorted(set(canonical_keys).difference(scales))
    if missing:
        raise ValueError(
            f"No complete in-sample seasonal scale for {len(missing)} evaluated "
            f"series (first: {missing[0]!r})"
        )

    grouped_errors = collections.defaultdict(lambda: [[], []])
    errors = pred_norm - targ_norm
    for key, error in zip(canonical_keys, errors):
        grouped_errors[key][0].append(error[0])
        grouped_errors[key][1].append(error[1])

    value_mase, weight_mase, value_rmsse, weight_rmsse = [], [], [], []
    for key, (value_errors, weight_errors) in grouped_errors.items():
        smae_v, smae_w, smse_v, smse_w = scales[key]
        if min(smae_v, smae_w, smse_v, smse_w) <= 1e-12:
            raise ValueError(f"Seasonal-naive scale is zero for evaluated series {key!r}")
        value_errors = np.asarray(value_errors)
        weight_errors = np.asarray(weight_errors)
        value_mase.append(float(np.mean(np.abs(value_errors)) / smae_v))
        weight_mase.append(float(np.mean(np.abs(weight_errors)) / smae_w))
        value_rmsse.append(float(np.sqrt(np.mean(value_errors ** 2) / smse_v)))
        weight_rmsse.append(float(np.sqrt(np.mean(weight_errors ** 2) / smse_w)))

    metrics["value_mase"] = float(np.mean(value_mase))
    metrics["weight_mase"] = float(np.mean(weight_mase))
    metrics["mase"] = (metrics["value_mase"] + metrics["weight_mase"]) / 2
    metrics["value_rmsse"] = float(np.mean(value_rmsse))
    metrics["weight_rmsse"] = float(np.mean(weight_rmsse))
    metrics["rmsse"] = (metrics["value_rmsse"] + metrics["weight_rmsse"]) / 2
