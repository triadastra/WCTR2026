#!/usr/bin/env python
"""
Run a non-learned baseline (seasonal_naive | random_walk | moving_average)
over the test split, with the SAME data pipeline, denormalization, and
metrics as scripts/inference.py — so baseline numbers are directly
comparable to the learned models.

Baselines have no learned parameters: this script evaluates them in one shot
and saves their definition and run contract instead of a weights checkpoint.

Usage:
    python scripts/run_baseline.py --model seasonal_naive
    python scripts/run_baseline.py --model moving_average --window 12
    python scripts/run_baseline.py --model random_walk --out-dir outputs/rw
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_data_config, load_config, set_seed, seed_worker
from src.data import TradeDataPipeline, DataConfig
from src.models import create_model
from src.inference.metrics import (
    compute_evaluation_metrics,
    compute_metrics,
    format_metrics,
)
from src.utils.provenance import cohort_provenance, data_protocol
from src.utils.run_contract import build_run_contract
from src.utils.run_artifacts import atomic_text, write_checkpoint_manifest

# Reuse the exact helpers used by the learned-model inference path.
from inference import (add_mase_rmsse, build_stats_lookup, denormalize,
                       run_inference, resolve_device)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a non-learned baseline")
    p.add_argument("--model", required=True,
                   choices=["seasonal_naive", "random_walk", "moving_average"])
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--out-dir", default=None,
                   help="Where to write metrics.json/predictions.csv "
                        "(default: outputs/<model>)")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override configured batch size; otherwise retain the experiment setting")
    p.add_argument("--window", type=int, default=12,
                   help="Moving-average window (ignored by other baselines)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cuda", "mps", "cpu"])
    p.add_argument("--aggregate", action="store_true",
                   help="Test 1 aggregated single series.")
    p.add_argument("--input-len", type=int, default=None,
                   help="Override input window length.")
    p.add_argument("--lag-count", type=int, default=None,
                   help="Override lag feature count.")
    p.add_argument("--train-target-start", default=None,
                   help="Inclusive first training-target month.")
    p.add_argument("--normalized", action="store_true",
                   help="Emit normalized predictions and shared normalized "
                        "scale-dependent metrics; MAPE/sMAPE use physical values.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print(f"Baseline: {args.model} (1-step walk-forward)")
    print("=" * 80)

    cfg = load_config(args.config)
    if args.aggregate and Path("config/exp1.yaml").exists():
        cfg = load_config("config/exp1.yaml", cfg)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.input_len is not None:
        if args.input_len <= 0:
            raise ValueError("--input-len must be positive")
        cfg["data"]["input_len"] = args.input_len
    if args.lag_count is not None:
        if args.lag_count < 0:
            raise ValueError("--lag-count must be non-negative")
        cfg["data"]["lag_count"] = args.lag_count
    if args.train_target_start is not None:
        cfg["data"]["train_target_start"] = args.train_target_start
    if args.window <= 0:
        raise ValueError("--window must be positive")
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    base_dir = Path(args.data_dir)
    from src.utils.config import resolve_data_paths
    cfg["data"] = resolve_data_paths(cfg["data"], base_dir)

    device = resolve_device(args.device, cfg)

    print("Loading and preprocessing data...")
    if args.aggregate:
        cfg["data"]["aggregate"] = True
    pipeline = TradeDataPipeline(DataConfig(**get_data_config(cfg)))
    pipeline.load_data()
    pipeline.create_splits()

    if args.normalized and not pipeline.config.normalize:
        raise ValueError(
            "--normalized requires data.normalize=true; otherwise the artifact "
            "would be mislabeled"
        )
    if args.model == "moving_average" and args.window > pipeline.config.input_len:
        raise ValueError(
            f"moving-average window {args.window} exceeds input_len "
            f"{pipeline.config.input_len}; the requested history is unavailable"
        )

    _, _, test_loader = pipeline.get_dataloaders(
        batch_size=cfg["training"]["batch_size"],
        num_workers=0,
        worker_init_fn=seed_worker,
        combo=False,
    )

    model_kwargs = {}
    if args.model == "moving_average":
        model_kwargs["window"] = args.window

    # Baseline behavior belongs in the persisted model contract. Keeping the
    # inherited neural architecture block here made two different moving-
    # average windows appear to be the same model configuration.
    cfg["model"] = {
        "name": args.model,
        "variant": "none",
        "window": args.window if args.model == "moving_average" else None,
    }

    model = create_model(
        name=args.model,
        variant="embeddings",  # irrelevant; baselines ignore encoding
        num_numeric_features=len(pipeline.feat_cols),
        num_states=len(pipeline.state2id),
        num_commodities=len(pipeline.comm2id),
        num_flows=len(pipeline.flow2id),
        **model_kwargs,
    ).to(device)

    id2state = {v: k for k, v in pipeline.state2id.items()}
    id2comm = {v: k for k, v in pipeline.comm2id.items()}
    id2flow = {v: k for k, v in pipeline.flow2id.items()}

    print("Running 1-step walk-forward over the test split...")
    pred_norm, targ_norm, keys, times = run_inference(
        model, test_loader, device, combo=False,
        id2state=id2state, id2comm=id2comm, id2flow=id2flow, id2combo=None,
    )
    print(f"  Collected {len(keys)} test predictions")

    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    targ_norm = np.asarray(targ_norm, dtype=np.float64)
    if args.normalized:
        maps = build_stats_lookup(pipeline.stats)
        pred_real = denormalize(pred_norm, keys, maps)
        targ_real = denormalize(targ_norm, keys, maps)
        metrics = compute_evaluation_metrics(
            pred_norm, targ_norm, pred_real, targ_real
        )
        # Same MASE/RMSSE as the learned models — baselines used to carry NaN
        # in these workbook columns because only inference.py computed them.
        add_mase_rmsse(metrics, pipeline, pred_norm, targ_norm, keys)
    elif pipeline.config.normalize:
        maps = build_stats_lookup(pipeline.stats)
        pred_real = denormalize(pred_norm, keys, maps)
        targ_real = denormalize(targ_norm, keys, maps)
        metrics = compute_metrics(pred_real, targ_real)
        add_mase_rmsse(metrics, pipeline, pred_norm, targ_norm, keys)
    else:
        pred_real = pred_norm.astype("float64")
        targ_real = targ_norm.astype("float64")
        metrics = compute_metrics(pred_real, targ_real)
        add_mase_rmsse(metrics, pipeline, pred_real, targ_real, keys)
    print()
    print(format_metrics(metrics))

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv + plots, same standardized layout as scripts/inference.py
    import csv
    import pandas as pd
    pred_path = out_dir / "predictions.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "State", "Commodity", "Import/Export", "Time",
            "pred_value", "pred_weight", "true_value", "true_weight",
            "pred_value_real", "pred_weight_real",
            "true_value_real", "true_weight_real",
        ])
        pred_export = pred_norm if args.normalized else pred_real
        targ_export = targ_norm if args.normalized else targ_real
        for i, k in enumerate(keys):
            w.writerow([
                k[0], k[1], k[2],
                pd.Timestamp(times[i]).strftime("%Y-%m-%d"),
                pred_export[i, 0], pred_export[i, 1],
                targ_export[i, 0], targ_export[i, 1],
                pred_real[i, 0], pred_real[i, 1],
                targ_real[i, 0], targ_real[i, 1],
            ])
    print(f"Predictions written to {pred_path}")

    # Keep metrics.json as the completion marker: publish it only after the
    # prediction table has been written successfully.
    run_contract = build_run_contract(
        cfg,
        model=args.model,
        variant="none",
        seed=args.seed,
        aggregate=args.aggregate,
        source_fingerprint=pipeline.source_fingerprint,
        source_domain=pipeline.source_domain,
    )
    atomic_text(out_dir / "baseline_definition.json", json.dumps({
        "kind": "parameter_free_baseline", "model_config": cfg["model"],
        "run_contract": run_contract,
    }, indent=2, allow_nan=False))
    write_checkpoint_manifest(out_dir, {"definition": "baseline_definition.json"},
                              kind="parameter_free_baseline", run_contract=run_contract)
    (out_dir / "metrics.json").write_text(json.dumps({
        "model": args.model,
        "variant": "none",
        "aggregate": bool(args.aggregate),
        "mode": "one_step_normalized" if args.normalized else "one_step",
        "prediction_units": "normalized" if args.normalized else "real",
        "scale_dependent_metric_units": "normalized" if args.normalized else "real",
        "percentage_metric_units": "real",
        "window": args.window if args.model == "moving_average" else None,
        "seed": args.seed,
        "n_samples": len(keys),
        "predictions_file": "predictions.csv",
        "source_fingerprint": pipeline.source_fingerprint,
        "source_domain": pipeline.source_domain,
        "run_contract": run_contract,
        "cohort": cohort_provenance(
            len(pipeline.kept_combos),
            bool(cfg["data"].get("cohort_from_train_only", True)),
        ),
        "data_protocol": data_protocol(
            cfg["data"]["input_len"], cfg["data"]["lag_count"],
            cfg["data"].get("train_target_start"),
        ),
        "metrics": metrics,
    }, indent=2, allow_nan=False))
    print(f"\nMetrics written to {out_dir / 'metrics.json'}")

    try:
        from src.inference.plotting import plot_run
        plot_run(pred_path, out_dir / "plots", title=args.model,
                 aggregate=args.aggregate)
    except Exception as e:
        print(f"[warn] plotting failed: {e}")


if __name__ == "__main__":
    main()
