#!/usr/bin/env python
"""
One-step walk-forward inference for time series forecasting models.

This is the *primary* evaluation path for the paper (1-step ahead). It:
  1. Rebuilds the data pipeline deterministically (same seed/config as
     training) so the test split, ID maps, and per-series normalization
     stats are bit-identical to what produced the checkpoint.
  2. Runs the model forward on the test split (input_len history -> next
     step), exactly one step ahead per window. No autoregression, no
     teacher forcing, no multi-horizon rollout -- those live in Phase 9.
  3. Denormalizes predictions AND targets back to real units using the
     4-level hierarchical backoff (exact -> commodity+flow -> commodity
     -> global), mirroring
     `paper included/shared_dataloader.py::get_series_stats_for_exact`.
  4. Keeps normalized scale-dependent metrics comparable across series while
     computing MAPE/sMAPE on denormalized physical values.

Usage:
    python scripts/inference.py --model lstm --variant embeddings \
        --checkpoint outputs/runs/best.pth

    python scripts/inference.py --model gru --variant cross_attention \
        --checkpoint outputs/runs/best.pth --batch-size 8

Multi-horizon (H-step direct, H-step AR) is intentionally NOT implemented
here yet; see plan.md Phase 9.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add repo root to path so ``from src.x import y`` resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_data_config, load_config, set_seed, seed_worker
from src.utils.config import apply_legacy_reproduction, require_legacy_output
from src.data import TradeDataPipeline, DataConfig
from src.inference.cohort import (
    apply_checkpoint_cohort_config,
    validate_checkpoint_cohort,
)
from src.utils.provenance import cohort_provenance, data_protocol
from src.models import create_model
from src.inference.metrics import (
    compute_evaluation_metrics,
    compute_metrics,
    format_metrics,
)
from src.utils.run_contract import (
    load_run_contract,
    validate_embedded_contract,
    validate_run_contract,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-step walk-forward inference / evaluation"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        # s4 = REAL S4 (src/models/s4.py) so the aggregate path works here; per-combo
        # Exp 2 S4 is still evaluated via external/s4 (s4_eval.py). There is no
        # s4d: it existed only inside the deleted ssm_models.py placeholder.
        choices=[
            "lstm", "gru", "transformer", "gpt", "transformer_linhead", "gpt_linhead",
            "xgboost", "lightgbm", "random_forest", "mamba", "mamba_nd", "s4", "s4_bidir",
        ],
        help="Model architecture (must match the checkpoint)",
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        choices=["onehot", "embeddings", "cross_attention", "film_attention", "identity_cross_attention",
                 "cross_attention_2d", "cross_attention_3d", "cross_attention_4d",
                 "matched_axial_2d", "matched_axial_3d", "matched_axial_4d"],
        help="Encoding variant (must match the checkpoint)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained checkpoint (best.pth / last.pth)",
    )
    parser.add_argument(
        "--config", type=str, default="config/base.yaml",
        help="Base configuration file (same as training)",
    )
    parser.add_argument(
        "--legacy-reproduction", action="store_true",
        help="Use the same pinned legacy data profile as training.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Where to write metrics.json + predictions.csv. "
             "Defaults to the checkpoint's directory.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=".",
        help="Base directory containing the data/ folder",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Test 1 aggregated single series (must match training).",
    )
    parser.add_argument(
        "--max-combo-plots", type=int, default=25,
        help="Per-combo benchmark: number of combos to plot SEPARATELY over the "
             "test horizon (-1 = every kept combo). Ignored under --aggregate.",
    )
    parser.add_argument(
        "--num-layers", type=int, default=None,
        help="Override model layers (must match training).",
    )
    parser.add_argument(
        "--input-len", type=int, default=None,
        help="Override input window length (must match training).",
    )
    parser.add_argument(
        "--lag-count", type=int, default=None,
        help="Override lag count (must match training).",
    )
    parser.add_argument(
        "--train-target-start", default=None,
        help="Inclusive first training-target month used by the trained protocol.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override eval batch size (combo models may need a small value)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the data pipeline (must match training for an "
             "identical test split)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "mps", "cpu"],
        help="Override device (default: from config, falling back to cpu)",
    )
    parser.add_argument(
        "--normalized", action="store_true",
        help="Emit normalized predictions and the shared normalized driving "
             "MSE/MAE/RMSE/R2 metrics. MAPE/sMAPE use denormalized physical "
             "values so percentages retain their meaning.",
    )
    parser.add_argument("--exp3", action="store_true", help="Use the Exp 3 head and completion policy")
    return parser.parse_args()


def resolve_device(requested, cfg):
    """Pick a torch.device, honoring an explicit request then config."""
    want = requested or cfg.get("training", {}).get("device", "cpu")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_stats_lookup(stats):
    """Build per-combo min/max lookup from pipeline stats.

    Returns exact_map: {(state, commodity, flow): (val_min, val_range, wt_min, wt_range)}
    """
    if not isinstance(stats, dict) or "exact" not in stats:
        raise ValueError("normalization stats are missing the exact-series table")
    exact_map = {}
    for _, r in stats["exact"].iterrows():
        key = (str(r["State"]), str(r["Commodity"]), str(r["Import/Export"]))
        val_range = float(r["val_max"]) - float(r["val_min"])
        wt_range  = float(r["wt_max"])  - float(r["wt_min"])
        values = (float(r["val_min"]), val_range, float(r["wt_min"]), wt_range)
        if not np.isfinite(values).all() or val_range < 0 or wt_range < 0:
            raise ValueError(f"invalid normalization statistics for series {key!r}")
        if key in exact_map:
            raise ValueError(f"duplicate normalization statistics for series {key!r}")
        exact_map[key] = (
            float(r["val_min"]), val_range if val_range != 0 else 1.0,
            float(r["wt_min"]),  wt_range  if wt_range  != 0 else 1.0,
        )
    return exact_map


def denormalize(norm_vw, keys, maps):
    """Inverse min-max normalization: x_real = x_norm * range + min.

    Args:
        norm_vw: (N, 2) normalized [value, weight]
        keys:    list of N (state, commodity, flow) string tuples
        maps:    output of build_stats_lookup
    Returns:
        (N, 2) denormalized [value, weight]
    """
    norm_vw = np.asarray(norm_vw, dtype=np.float64)
    if norm_vw.ndim != 2 or norm_vw.shape[1] != 2:
        raise ValueError("denormalize expects an (n_samples, 2) array")
    if len(keys) != len(norm_vw):
        raise ValueError("denormalize key count does not match prediction count")
    if not np.isfinite(norm_vw).all():
        raise ValueError("denormalize input contains non-finite values")
    out = np.empty_like(norm_vw, dtype=np.float64)
    for i, key in enumerate(keys):
        if key not in maps:
            raise ValueError(f"missing normalization statistics for series {key!r}")
        val_min, val_range, wt_min, wt_range = maps[key]
        out[i, 0] = norm_vw[i, 0] * val_range + val_min
        out[i, 1] = norm_vw[i, 1] * wt_range  + wt_min
    return out


from src.inference.scaled_metrics import add_mase_rmsse  # noqa: E402


def run_inference(model, test_loader, device, combo, id2state, id2comm, id2flow,
                  id2combo):
    """Walk the test split one step ahead, collecting normalized
    predictions/targets plus the (state, commodity, flow) key per sample.

    Returns (pred_norm, targ_norm, keys, times) where pred/targ are (N, 2)
    arrays in normalized space, keys is a list of N string tuples and times
    is a list of N pd.Timestamps (the target month of each prediction).
    """
    model.eval()
    preds, targs, keys, times = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x_numeric = batch["x_numeric"].to(device)

            if combo:
                group_mask = batch["group_mask"].to(device)
                # (B, G, 2)
                out = model(x_numeric, None, None, None, group_mask=group_mask)
                tv = batch["target_value"]   # (B, G)
                tw = batch["target_weight"]  # (B, G)
                mask = batch["group_mask"].bool().cpu().numpy()  # (B, G)

                out_np = out.cpu().numpy()
                tv_np = tv.cpu().numpy()
                tw_np = tw.cpu().numpy()
                batch_times = batch["target_time"]
                B, G = mask.shape
                for b in range(B):
                    for g in range(G):
                        if not mask[b, g]:
                            continue  # combo absent at this target time
                        state, commodity, flow = id2combo[g]
                        preds.append((out_np[b, g, 0], out_np[b, g, 1]))
                        targs.append((tv_np[b, g], tw_np[b, g]))
                        keys.append((str(state), str(commodity), str(flow)))
                        times.append(batch_times[b])
            else:
                state_ids = batch["state_ids"].to(device)
                comm_ids = batch["comm_ids"].to(device)
                flow_ids = batch["flow_ids"].to(device)
                out = model(x_numeric, state_ids, comm_ids, flow_ids)  # (B, 2)

                out_np = out.cpu().numpy()
                tv_np = batch["target_value"].cpu().numpy()
                tw_np = batch["target_weight"].cpu().numpy()
                s_np = batch["state_ids"].cpu().numpy()
                c_np = batch["comm_ids"].cpu().numpy()
                f_np = batch["flow_ids"].cpu().numpy()
                batch_times = batch["target_time"]
                for b in range(out_np.shape[0]):
                    preds.append((out_np[b, 0], out_np[b, 1]))
                    targs.append((tv_np[b], tw_np[b]))
                    keys.append((
                        str(id2state[int(s_np[b])]),
                        str(id2comm[int(c_np[b])]),
                        str(id2flow[int(f_np[b])]),
                    ))
                    times.append(batch_times[b])

    return np.asarray(preds, dtype=np.float64), np.asarray(targs, dtype=np.float64), keys, times


def collect_keys_times(test_loader, id2state, id2comm, id2flow):
    """Walk the (deterministic, unshuffled) test loader collecting the
    (state, commodity, flow) key and target month per sample, in loader
    order. Used for tree models whose predict path consumes the loader
    separately via extract_tabular_xy (same order)."""
    keys, times = [], []
    for batch in test_loader:
        s_np = batch["state_ids"].cpu().numpy()
        c_np = batch["comm_ids"].cpu().numpy()
        f_np = batch["flow_ids"].cpu().numpy()
        batch_times = batch["target_time"]
        for b in range(len(s_np)):
            keys.append((
                str(id2state[int(s_np[b])]),
                str(id2comm[int(c_np[b])]),
                str(id2flow[int(f_np[b])]),
            ))
            times.append(batch_times[b])
    return keys, times


def load_state_dict_flex(model, checkpoint_path, device):
    """Load a checkpoint that may be a raw state_dict (best.pth) or a
    wrapped dict with 'model_state_dict' (best.pth / last.pth).

    Returns the embedded run contract, if present.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        embedded_contract = ckpt.get("run_contract")
    else:
        state = ckpt
        embedded_contract = None
    model.load_state_dict(state)
    return embedded_contract


def main():
    args = parse_args()

    # Seed BEFORE building the pipeline so the test split / stats / id maps
    # match training bit-for-bit.
    set_seed(args.seed)

    print("=" * 80)
    print(f"Inference: {args.model} / {args.variant} (1-step walk-forward)")
    print("=" * 80)

    # Resolve config exactly like train.py: base -> model -> variant.
    cfg = load_config(args.config)
    model_config_path = f"config/models/{args.model}.yaml"
    if Path(model_config_path).exists():
        cfg = load_config(model_config_path, cfg)
    variant_config_path = f"config/variants/{args.variant}.yaml"
    if Path(variant_config_path).exists():
        cfg = load_config(variant_config_path, cfg)

    if args.exp3:
        if args.aggregate or args.legacy_reproduction or args.variant != "embeddings":
            raise ValueError("--exp3 requires non-aggregate embeddings without legacy reproduction")
        from src.utils.exp3_protocol import apply_exp3_config
        cfg = apply_exp3_config(cfg, args.model)

    # Same Exp 1 overrides as train.py, so config resolution stays identical.
    if args.aggregate and Path("config/exp1.yaml").exists():
        cfg = load_config("config/exp1.yaml", cfg)

    if args.legacy_reproduction:
        cfg = apply_legacy_reproduction(cfg, aggregate=args.aggregate)
        require_legacy_output(args.out_dir or str(Path(args.checkpoint).parent))

    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        cfg["training"]["batch_size"] = args.batch_size

    base_dir = Path(args.data_dir)
    if args.aggregate:
        cfg["data"]["aggregate"] = True
    if args.num_layers is not None:
        if args.num_layers <= 0:
            raise ValueError("--num-layers must be positive")
        layer_key = "n_layers" if args.model in ("mamba", "mamba_nd", "s4", "s4_bidir") else "num_layers"
        cfg["model"][layer_key] = args.num_layers
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

    if args.model in ("xgboost", "lightgbm", "random_forest"):
        cfg["model"]["random_state"] = int(args.seed)

    run_contract = load_run_contract(args.checkpoint)
    training_completion = None
    if args.exp3:
        from src.utils.training_completion import read_completion
        training_completion = read_completion(args.checkpoint, run_contract)
    else:
        from src.utils.exp1_completion import read_exp1_completion
        # Allow a completed smoke run to be inspected. The canonical paper
        # gate separately requires early stopping for neural arms.
        training_completion = read_exp1_completion(
            args.checkpoint, run_contract, require_early_stopping=False
        )

    # A checkpoint must be evaluated on the same series population it was
    # trained on. New artifacts record the cohort mode explicitly; artifacts
    # from before that field existed are legacy full-timeline cohorts.
    cohort_metadata = apply_checkpoint_cohort_config(cfg, args.checkpoint)
    if cohort_metadata is not None:
        cohort_mode = (
            "train-only" if cfg["data"]["cohort_from_train_only"]
            else "legacy full-timeline"
        )
        print(f"Checkpoint cohort: {cohort_mode} "
              f"({cohort_metadata.get('num_combos', 'unknown')} combos)")

    from src.utils.config import resolve_data_paths
    cfg["data"] = resolve_data_paths(cfg["data"], base_dir)

    device = resolve_device(args.device, cfg)
    print(f"Device: {device}")

    # Build pipeline deterministically.
    print("Loading and preprocessing data...")
    data_config = DataConfig(**get_data_config(cfg))
    pipeline = TradeDataPipeline(data_config)
    pipeline.load_data()
    pipeline.create_splits()
    validate_checkpoint_cohort(
        pipeline.kept_combos,
        cohort_metadata,
        source_fingerprint=pipeline.source_fingerprint,
    )
    validate_run_contract(
        run_contract,
        cfg,
        model=args.model,
        variant=args.variant,
        seed=args.seed,
        aggregate=args.aggregate,
        source_fingerprint=pipeline.source_fingerprint,
        source_domain=pipeline.source_domain,
    )

    if args.normalized and not pipeline.config.normalize:
        raise ValueError(
            "--normalized requires data.normalize=true; otherwise the pipeline "
            "contains real-unit values and the artifact would be mislabeled"
        )

    _axial = ("cross_attention_2d", "cross_attention_3d", "cross_attention_4d")
    _matched_axial = ("matched_axial_2d", "matched_axial_3d", "matched_axial_4d")
    _all_axial = _axial + _matched_axial
    combo = (
        (args.model in ("lstm", "gru")
         and args.variant in ("cross_attention", "film_attention", "identity_cross_attention") + _all_axial)
        or (args.model in ("transformer", "gpt", "transformer_linhead", "gpt_linhead",
                            "mamba", "s4", "s4_bidir")
            and args.variant in _all_axial)
        or (args.model == "mamba_nd" and args.variant in _all_axial)
    )

    # Combo models are per-time-step over the full (B, L, G, F) panel, so the
    # eval batch must stay small -- especially mamba_nd, whose Mamba-2 Triton
    # kernel hits an illegal memory access if the folded batch is large. Cap it
    # unless the user gave an explicit --batch-size.
    _COMBO_EVAL_BS = {"mamba_nd": 2, "gru": 2, "lstm": 4, "transformer": 4, "gpt": 4,
                      "transformer_linhead": 4, "gpt_linhead": 4}
    if combo and not args.batch_size:
        cfg["training"]["batch_size"] = _COMBO_EVAL_BS.get(args.model, 2)
        print(f"  Combo eval: capping batch to {cfg['training']['batch_size']} "
              f"for ({args.model}). Override with --batch-size.")

    _, _, test_loader = pipeline.get_dataloaders(
        batch_size=cfg["training"]["batch_size"],
        num_workers=0,
        worker_init_fn=seed_worker,
        combo=combo,
    )

    # Build the model exactly as train.py did.
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model_cfg.pop("variant", None)

    # Same translation as train.py: config embedding_dims -> BaseModel kwargs.
    emb_dims = model_cfg.pop("embedding_dims", None)
    if emb_dims:
        model_cfg.setdefault("state_embed_dim", emb_dims.get("state", 8))
        model_cfg.setdefault("comm_embed_dim", emb_dims.get("commodity", 32))
        model_cfg.setdefault("flow_embed_dim", emb_dims.get("flow", 2))

    # Mirror train.py: the tree arms take their RNG seed as a constructor arg.
    # Predictions come from the loaded checkpoint, so this does not change any
    # number here -- it keeps construction identical to training so a config
    # carrying `random_state` cannot diverge between the two paths.
    if args.model in ("xgboost", "lightgbm", "random_forest"):
        model_cfg.setdefault("random_state", args.seed)

    num_states = len(pipeline.state2id)
    num_commodities = len(pipeline.comm2id)
    num_flows = len(pipeline.flow2id)
    num_numeric_features = len(pipeline.feat_cols)

    if combo:
        model_cfg["num_combos"] = pipeline.num_combos
        model_cfg["features_per_group"] = num_numeric_features
        # Axial combo variants need the (S,C,Flow) lattice + per-combo coords.
        model_cfg["combo_coords"] = pipeline.combo_coords
        model_cfg["lattice_dims"] = pipeline.lattice_dims

    model = create_model(
        name=args.model,
        variant=args.variant,
        num_numeric_features=num_numeric_features,
        num_states=num_states,
        num_commodities=num_commodities,
        num_flows=num_flows,
        **model_cfg,
    )
    if args.model in ("xgboost", "lightgbm", "random_forest"):
        # Tree models save with their native format (best_xgb / best_lgb),
        # not a torch state_dict.
        model.load(args.checkpoint)
        embedded_contract = getattr(model, "run_contract", None)
    else:
        embedded_contract = load_state_dict_flex(model, args.checkpoint, device)
    validate_embedded_contract(
        embedded_contract,
        run_contract,
        checkpoint_path=args.checkpoint,
    )
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Inverse ID maps for denormalization key reconstruction.
    id2state = {v: k for k, v in pipeline.state2id.items()}
    id2comm = {v: k for k, v in pipeline.comm2id.items()}
    id2flow = {v: k for k, v in pipeline.flow2id.items()}
    id2combo = getattr(pipeline, "id2combo", None)

    print("Running 1-step walk-forward over the test split...")
    if args.model in ("xgboost", "lightgbm", "random_forest"):
        # Tree models predict from tabular features, not forward().
        pred_norm, targ_norm = model.predict_loader(test_loader)
        keys, times = collect_keys_times(test_loader, id2state, id2comm, id2flow)
    else:
        pred_norm, targ_norm, keys, times = run_inference(
            model, test_loader, device, combo,
            id2state, id2comm, id2flow, id2combo,
        )
    print(f"  Collected {len(keys)} test predictions")

    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    targ_norm = np.asarray(targ_norm, dtype=np.float64)
    if args.normalized:
        # The artifact's primary columns, driving loss, and scale-dependent
        # metrics stay in the common per-series normalized domain. MAPE and
        # sMAPE must use physical values because min-max translation changes a
        # percentage error.
        maps = build_stats_lookup(pipeline.stats)
        pred_real = denormalize(pred_norm, keys, maps)
        targ_real = denormalize(targ_norm, keys, maps)
        metrics = compute_evaluation_metrics(
            pred_norm, targ_norm, pred_real, targ_real
        )
        add_mase_rmsse(metrics, pipeline, pred_norm, targ_norm, keys)
    elif pipeline.config.normalize:
        # Denormalize both predictions and targets to real units BEFORE metrics.
        maps = build_stats_lookup(pipeline.stats)
        pred_real = denormalize(pred_norm, keys, maps)
        targ_real = denormalize(targ_norm, keys, maps)
        metrics = compute_metrics(pred_real, targ_real)
        add_mase_rmsse(metrics, pipeline, pred_norm, targ_norm, keys)
    else:
        # The pipeline already produced real units.  Applying inverse min-max
        # here would scale them a second time by billions for Exp 1.
        pred_real = np.asarray(pred_norm, dtype=np.float64)
        targ_real = np.asarray(targ_norm, dtype=np.float64)
        metrics = compute_metrics(pred_real, targ_real)
        add_mase_rmsse(metrics, pipeline, pred_real, targ_real, keys)
    # Parameter count (transparency for the capacity-match question).
    try:
        # BaseModel.get_num_params() counts torch Parameters. Native tree
        # estimators live in joblib/booster objects, so reporting zero here
        # falsely describes a fitted forest as parameter-free. Their estimator
        # and node/leaf complexity belongs in native model metadata instead.
        if (
            args.model not in ("xgboost", "lightgbm", "random_forest")
            and hasattr(model, "get_num_params")
        ):
            metrics["num_params"] = int(model.get_num_params())
    except Exception:
        pass
    print()
    print(format_metrics(metrics))

    # Write outputs next to the checkpoint (or to --out-dir).
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # The primary value columns follow prediction_units.  Normalized benchmark
    # artifacts also carry the physical values used for MAPE/sMAPE, allowing
    # provenance validation to bind both metric domains to the exact prediction
    # bytes rather than trusting metrics.json.
    pred_path = out_dir / "predictions.csv"
    states = [k[0] for k in keys]
    comms = [k[1] for k in keys]
    flows = [k[2] for k in keys]
    import csv
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
        for i in range(len(keys)):
            w.writerow([
                states[i], comms[i], flows[i],
                pd.Timestamp(times[i]).strftime("%Y-%m-%d"),
                pred_export[i, 0], pred_export[i, 1],
                targ_export[i, 0], targ_export[i, 1],
                pred_real[i, 0], pred_real[i, 1],
                targ_real[i, 0], targ_real[i, 1],
            ])
    print(f"Predictions written to {pred_path}")

    # metrics.json is the run-completion marker used by the resumable
    # launchers. Write it only after the required predictions artifact closes
    # successfully, so an interrupted export cannot be mistaken for a current
    # complete run.
    metrics_path = out_dir / "metrics.json"
    contract_for_output = {
        key: value for key, value in run_contract.items()
        if not str(key).startswith("_")
    }
    from src.utils.model_measurements import model_measurements
    metrics_payload = {
        "model": args.model,
        "variant": args.variant,
        "aggregate": bool(args.aggregate),
        "mode": "one_step_normalized" if args.normalized else "one_step",
        "prediction_units": "normalized" if args.normalized else "real",
        "scale_dependent_metric_units": "normalized" if args.normalized else "real",
        "percentage_metric_units": "real",
        "seed": args.seed,
        "n_samples": len(keys),
        "predictions_file": "predictions.csv",
        "source_fingerprint": pipeline.source_fingerprint,
        "source_domain": pipeline.source_domain,
        "run_contract": contract_for_output,
        "training_completion": training_completion,
        "cohort": cohort_provenance(
            len(pipeline.kept_combos),
            bool(cfg["data"].get("cohort_from_train_only", True)),
        ),
        "data_protocol": data_protocol(
            cfg["data"]["input_len"], cfg["data"]["lag_count"],
            cfg["data"].get("train_target_start"),
        ),
        "checkpoint": str(args.checkpoint),
        # Causality / encoder revision of the network that produced these
        # numbers, so the results table can report "two-sided over the window"
        # next to each arm instead of inferring it from the YAML.
        "model_measurements": model_measurements(model, args.model),
        "architecture": (model.architecture_provenance()
                         if hasattr(model, "architecture_provenance") else {}),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, allow_nan=False))
    print(f"\nMetrics written to {metrics_path}")

    # Plots over the full test horizon: per-combo (separate) for the benchmark,
    # single-series for the aggregated test. Written to <out_dir>/plots/.
    try:
        from src.inference.plotting import plot_run
        plot_run(pred_path, out_dir / "plots",
                 title=f"{args.model}/{args.variant}",
                 max_combo_plots=args.max_combo_plots, aggregate=args.aggregate)
    except Exception as e:  # plotting must never fail the eval
        print(f"[warn] plotting failed: {e}")


if __name__ == "__main__":
    main()
