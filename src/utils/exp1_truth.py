"""Bind Exp1 prediction targets and units to the pinned original CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data.dataloader import DataConfig, TradeDataPipeline
from src.utils.config import get_data_config, load_config
from src.utils.experiment_manifest import (
    exp1_source_domain, load_manifest, validate_exp1_config,
)


def load_exp1_truth(repo: Path, data_root: Path | None = None) -> dict:
    """Rebuild only the aggregate panel; never trust run-local saved targets.

    No network access, source writes, or model execution. Call once per
    collector invocation, not once per seed. The loader hashes consumed bytes.
    """
    repo = Path(repo)
    data_root = repo if data_root is None else Path(data_root)
    manifest = load_manifest(repo / "config/experiment_manifest.env")
    cfg = load_config(str(repo / "config/base.yaml"))
    cfg = load_config(str(repo / "config/exp1.yaml"), cfg)
    validate_exp1_config(manifest, cfg)
    data = get_data_config(cfg)
    data.update(aggregate=True, imports_dir=str(data_root / "data/imports"),
                exports_dir=str(data_root / "data/exports"))
    from src.utils.config import resolve_data_paths
    data = resolve_data_paths(data, data_root)
    pipe = TradeDataPipeline(DataConfig(**data))
    pipe.load_data()
    if pipe.source_domain != exp1_source_domain(manifest):
        raise ValueError("reconstructed Exp1 source domain differs from manifest")
    pipe.create_splits()
    stats = pipe.stats["exact"]
    if len(stats) != 1:
        raise ValueError("Exp1 truth must have exactly one normalization series")
    row = stats.iloc[0]
    minimum = np.array([row.val_min, row.wt_min], dtype=np.float64)
    scale = np.array([row.val_max - row.val_min, row.wt_max - row.wt_min], dtype=np.float64)
    scale[scale == 0] = 1.0
    targets = {}
    for _, row in pipe.df_test.iterrows():
        key = (str(row.State), str(row.Commodity), str(row["Import/Export"]),
               row.Time.strftime("%Y-%m-%d"))
        # Dataset tensors carry float32 targets. The evaluator promotes those
        # tensors before inverse scaling; reproduce that same rounding path.
        normalized = np.array([row.Value, row.Weight], dtype=np.float32).astype(np.float64)
        targets[key] = np.concatenate([normalized, normalized * scale + minimum])
    if len(targets) != 5:
        raise ValueError("Exp1 source reconstruction must yield five test targets")
    train = pipe.df_metric_grid[
        pipe.df_metric_grid.Time <= pipe.config.train_end
    ].sort_values("Time")
    differences = train[["Value", "Weight"]].diff(12).dropna().to_numpy(dtype=np.float64)
    seasonal_mae = np.mean(np.abs(differences), axis=0)
    seasonal_mse = np.mean(differences ** 2, axis=0)
    if not np.isfinite([seasonal_mae, seasonal_mse]).all() or (
        seasonal_mae <= 0
    ).any() or (seasonal_mse <= 0).any():
        raise ValueError("invalid source-derived seasonal metric scales")
    return {"targets": targets, "minimum": minimum, "scale": scale,
            "seasonal_mae": seasonal_mae, "seasonal_mse": seasonal_mse}


def validate_exp1_truth(payload: dict, metrics_path: Path, truth: dict) -> tuple[bool, str]:
    """Check targets and prediction inverse transforms, not just metric agreement."""
    name = payload.get("predictions_file")
    if name not in {"predictions.csv", "predictions_norm.csv"}:
        return False, "missing canonical prediction file"
    seen = set()
    errors = []
    try:
        with (Path(metrics_path).parent / name).open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = tuple(row[c] for c in ("State", "Commodity", "Import/Export", "Time"))
                if key in seen or key not in truth["targets"]:
                    return False, "prediction identities differ from pinned source targets"
                seen.add(key)
                actual = np.array([float(row[c]) for c in (
                    "true_value", "true_weight", "true_value_real", "true_weight_real",
                )])
                if not np.allclose(actual, truth["targets"][key], rtol=1e-9, atol=1e-8):
                    return False, f"test targets differ from pinned source at {key}"
                pred = np.array([float(row["pred_value"]), float(row["pred_weight"])])
                errors.append(pred - truth["targets"][key][:2])
                real = np.array([float(row["pred_value_real"]), float(row["pred_weight_real"])])
                expected = pred * truth["scale"] + truth["minimum"]
                if not np.isfinite(pred).all() or not np.isfinite(real).all() or not np.allclose(
                    real, expected, rtol=1e-9, atol=1e-8
                ):
                    return False, f"prediction units differ from pinned source scaling at {key}"
    except (OSError, KeyError, ValueError, TypeError, csv.Error) as exc:
        return False, f"cannot validate pinned source targets: {exc}"
    if seen != set(truth["targets"]):
        return False, "prediction target population differs from pinned source"
    errors = np.asarray(errors)
    mase = np.mean(np.abs(errors), axis=0) / truth["seasonal_mae"]
    rmsse = np.sqrt(np.mean(errors ** 2, axis=0) / truth["seasonal_mse"])
    expected_metrics = {
        "value_mase": mase[0], "weight_mase": mase[1], "mase": np.mean(mase),
        "value_rmsse": rmsse[0], "weight_rmsse": rmsse[1], "rmsse": np.mean(rmsse),
    }
    for name, expected in expected_metrics.items():
        try:
            actual = float(payload["metrics"][name])
        except (KeyError, ValueError, TypeError):
            return False, f"missing or invalid source-scaled metric {name}"
        if not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-6, atol=1e-8):
            return False, f"metric {name} differs from pinned source seasonal scaling"
    return True, "pinned source targets and scaling verified"
