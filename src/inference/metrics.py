"""
Evaluation metrics for time series forecasting.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Canonical metric schema for every benchmark artifact.  Keep this ordered:
# result tables use the same sequence and provenance validation uses the same
# names, so an evaluator cannot silently publish a partial metric suite that a
# launcher later mistakes for a complete run.
SCALE_DEPENDENT_METRIC_NAMES = (
    "value_mse", "weight_mse",
    "mae", "value_mae", "weight_mae",
    "rmse", "value_rmse", "weight_rmse",
    "r2", "value_r2", "weight_r2",
)
PERCENTAGE_METRIC_NAMES = (
    "value_mape", "weight_mape",
    "smape", "value_smape", "weight_smape",
)
DIRECT_METRIC_NAMES = (
    *SCALE_DEPENDENT_METRIC_NAMES,
    *PERCENTAGE_METRIC_NAMES,
)
SCALED_METRIC_NAMES = (
    "mase", "value_mase", "weight_mase",
    "rmsse", "value_rmsse", "weight_rmsse",
)
REQUIRED_EVALUATION_METRICS = (
    "norm_mse_combined",
    *DIRECT_METRIC_NAMES,
    *SCALED_METRIC_NAMES,
)


def symmetric_mape(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Channel-level symmetric MAPE on the conventional 0--200% scale."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("symmetric_mape expects non-empty arrays with matching shapes")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("symmetric_mape inputs must be finite")
    denom = np.abs(targets) + np.abs(predictions)
    ok = denom > 1e-12
    # The continuous convention for an exactly correct 0/0 term is zero error.
    # Excluding those rows biased sparse-series sMAPE upward, while returning
    # NaN for an all-zero channel poisoned otherwise valid JSON artifacts.
    terms = np.zeros_like(denom, dtype=np.float64)
    terms[ok] = 2.0 * np.abs(predictions[ok] - targets[ok]) / denom[ok]
    return float(np.mean(terms) * 100)


def combined_smape(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean of value-channel and weight-channel sMAPE.

    Keeping the channels separate prevents the channel with larger denominators
    from implicitly receiving more weight.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.ndim != 2 or targets.shape != predictions.shape or predictions.shape[1] != 2:
        raise ValueError("combined_smape expects matching (n_samples, 2) arrays")
    return float(np.nanmean([
        symmetric_mape(targets[:, 0], predictions[:, 0]),
        symmetric_mape(targets[:, 1], predictions[:, 1]),
    ]))


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mape_floor: float = 1.0,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: (n_samples, 2) - [value, weight]
        targets: (n_samples, 2) - [value, weight]
        mape_floor: MAPE is undefined as the true value approaches zero.
            Per the pipeline contract, metrics are computed on
            DENORMALIZED real-unit (dollar) values, so targets with
            |true| <= ``mape_floor`` are excluded from MAPE only. The
            default of 1.0 drops sub-$1 "trades" — which are numerically
            zero (a no-trade month denormalizes to ~$0 with float
            residual) and otherwise make MAPE explode to billions of %.
            MAE/RMSE/R2 use all samples. MAPE is stable for any floor in
            roughly [1, 1000] on this data.

    Returns:
        Dictionary with metrics for value and weight
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if (
        predictions.ndim != 2
        or predictions.shape[1:] != (2,)
        or targets.shape != predictions.shape
        or len(predictions) == 0
    ):
        raise ValueError("compute_metrics expects matching non-empty (n_samples, 2) arrays")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("compute_metrics inputs must contain only finite values")
    if not np.isfinite(mape_floor) or mape_floor < 0:
        raise ValueError("mape_floor must be finite and non-negative")

    metrics = {}

    # Separate value and weight
    pred_value = predictions[:, 0]
    pred_weight = predictions[:, 1]
    targ_value = targets[:, 0]
    targ_weight = targets[:, 1]

    # Value metrics
    metrics["value_mae"] = mean_absolute_error(targ_value, pred_value)
    metrics["value_mse"] = mean_squared_error(targ_value, pred_value)
    metrics["value_rmse"] = np.sqrt(metrics["value_mse"])

    # Avoid division by ~zero in MAPE (see mape_floor docstring).
    mask_value = np.abs(targ_value) > mape_floor
    if mask_value.sum() > 0:
        metrics["value_mape"] = np.mean(
            np.abs((targ_value[mask_value] - pred_value[mask_value]) / targ_value[mask_value])
        ) * 100
    else:
        metrics["value_mape"] = np.nan

    metrics["value_r2"] = r2_score(targ_value, pred_value)

    # Weight metrics
    metrics["weight_mae"] = mean_absolute_error(targ_weight, pred_weight)
    metrics["weight_mse"] = mean_squared_error(targ_weight, pred_weight)
    metrics["weight_rmse"] = np.sqrt(metrics["weight_mse"])

    mask_weight = np.abs(targ_weight) > mape_floor
    if mask_weight.sum() > 0:
        metrics["weight_mape"] = np.mean(
            np.abs((targ_weight[mask_weight] - pred_weight[mask_weight]) / targ_weight[mask_weight])
        ) * 100
    else:
        metrics["weight_mape"] = np.nan

    metrics["weight_r2"] = r2_score(targ_weight, pred_weight)

    # Overall metrics (average)
    metrics["mae"] = (metrics["value_mae"] + metrics["weight_mae"]) / 2
    metrics["rmse"] = (metrics["value_rmse"] + metrics["weight_rmse"]) / 2
    metrics["r2"] = (metrics["value_r2"] + metrics["weight_r2"]) / 2

    # ---- Scale-free metric --------------------------------------------------
    # MASE/RMSSE require series identities so each series is scaled before the
    # cross-series average. They are added by inference.add_mase_rmsse, which
    # has those keys; accepting anonymous per-point scales here previously
    # implemented a different, point-weighted metric under the same name.
    metrics["value_smape"] = symmetric_mape(targ_value, pred_value)
    metrics["weight_smape"] = symmetric_mape(targ_weight, pred_weight)
    metrics["smape"] = float(np.nanmean([metrics["value_smape"], metrics["weight_smape"]]))

    return metrics


def compute_evaluation_metrics(
    predictions_normalized: np.ndarray,
    targets_normalized: np.ndarray,
    predictions_real: np.ndarray,
    targets_real: np.ndarray,
    mape_floor: float = 1.0,
) -> Dict[str, float]:
    """Compute normalized losses plus physically meaningful percentages.

    Min-max normalization includes a per-series translation.  Percentage
    errors (MAPE and sMAPE) are not invariant to that translation, so computing
    them in normalized space changes their meaning.  The one shared selection
    metric remains MSE over the flattened normalized value/weight panel.
    MAE/MSE/RMSE and R2 also stay in the normalized domain so USD and kg are
    not pooled and high-volume series do not dominate; only MAPE/sMAPE come
    from the physical USD/kg arrays.

    MASE/RMSSE are intentionally added separately by the inference harness,
    because they require keyed in-sample seasonal scales rather than only the
    prediction arrays.
    """
    predictions_normalized = np.asarray(predictions_normalized, dtype=np.float64)
    targets_normalized = np.asarray(targets_normalized, dtype=np.float64)
    if (
        predictions_normalized.ndim != 2
        or predictions_normalized.shape[1:] != (2,)
        or targets_normalized.shape != predictions_normalized.shape
        or len(predictions_normalized) == 0
    ):
        raise ValueError(
            "compute_evaluation_metrics expects matching non-empty normalized "
            "(n_samples, 2) arrays"
        )
    if not np.isfinite(predictions_normalized).all() or not np.isfinite(
        targets_normalized
    ).all():
        raise ValueError("normalized evaluation inputs must contain only finite values")

    predictions_real = np.asarray(predictions_real, dtype=np.float64)
    targets_real = np.asarray(targets_real, dtype=np.float64)
    if (
        predictions_real.shape != predictions_normalized.shape
        or targets_real.shape != targets_normalized.shape
    ):
        raise ValueError(
            "real and normalized evaluation arrays must describe the same "
            "(n_samples, 2) population"
        )

    normalized_metrics = compute_metrics(
        predictions_normalized, targets_normalized, mape_floor=0.0
    )
    real_metrics = compute_metrics(predictions_real, targets_real, mape_floor=mape_floor)
    metrics = {
        name: normalized_metrics[name] for name in SCALE_DEPENDENT_METRIC_NAMES
    }
    metrics.update({name: real_metrics[name] for name in PERCENTAGE_METRIC_NAMES})
    metrics["norm_mse_combined"] = float(
        np.mean((predictions_normalized - targets_normalized) ** 2)
    )
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics for printing.

    Args:
        metrics: Metrics dictionary

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Evaluation Metrics")
    lines.append("=" * 60)

    lines.append("\nValue Metrics:")
    lines.append(f"  MAE:  {metrics['value_mae']:>10.4f}")
    lines.append(f"  RMSE: {metrics['value_rmse']:>10.4f}")
    lines.append(f"  MAPE: {metrics['value_mape']:>10.2f}%")
    lines.append(f"  R²:   {metrics['value_r2']:>10.4f}")

    lines.append("\nWeight Metrics:")
    lines.append(f"  MAE:  {metrics['weight_mae']:>10.4f}")
    lines.append(f"  RMSE: {metrics['weight_rmse']:>10.4f}")
    lines.append(f"  MAPE: {metrics['weight_mape']:>10.2f}%")
    lines.append(f"  R²:   {metrics['weight_r2']:>10.4f}")

    lines.append("\nOverall Metrics:")
    lines.append(f"  MAE:  {metrics['mae']:>10.4f}")
    lines.append(f"  RMSE: {metrics['rmse']:>10.4f}")
    lines.append(f"  R²:   {metrics['r2']:>10.4f}")

    lines.append("=" * 60)

    return "\n".join(lines)


def save_metrics(
    metrics: Dict[str, float],
    save_path: str,
    model_name: str,
    variant: str,
) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Metrics dictionary
        save_path: Path to save
        model_name: Model name
        variant: Variant name
    """
    import json

    output = {
        "model": model_name,
        "variant": variant,
        "metrics": metrics,
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Metrics saved to {save_path}")
