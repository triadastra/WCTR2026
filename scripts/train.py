#!/usr/bin/env python
"""
Training script for time series forecasting models.

Usage:
    python scripts/train.py --model lstm --variant embeddings
    python scripts/train.py --model gru --variant film_attention --config config/base.yaml
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path so ``from src.x import y`` resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_data_config, load_config, set_seed, seed_worker
from src.utils.run_contract import build_run_contract, save_run_contract
from src.data import TradeDataPipeline, DataConfig
from src.utils.config import apply_legacy_reproduction, require_legacy_output
from src.models import create_model
from src.training import Trainer


# Per-config combo batch sizes used by the paper. The combo path
# (cross_attention / film_attention for lstm/gru) builds (B, L, G, F)
# windows over ALL groups at once, so its memory footprint is far larger
# than the per-series path and it OOMs at the per-series default of 64.
# The paper did NOT use a single combo batch size -- each combo config
# used its own value (paper included/{LSTM,GRU}/{LSTM,GRU}_combo*.py). We
# reproduce those exactly so default runs match the paper bit-for-bit.
# Override with --batch-size for smoke tests (e.g. 8 fits the 5090).
COMBO_BATCH_SIZES = {
    ("lstm", "cross_attention"): 1,    # was 512 (per-series), but ComboDataset is per-time-step (~156 train steps)
    ("lstm", "film_attention"): 1,     # LSTM/LSTM_combo.py
    ("gru", "cross_attention"): 1,     # GRU/GRU_combo_caonly.py
    ("gru", "film_attention"): 1,      # GRU/GRU_combo.py
    # Axial (S4ND-style) combo variants: same per-time-step combo dataset.
    ("lstm", "cross_attention_2d"): 1, ("lstm", "cross_attention_3d"): 1, ("lstm", "cross_attention_4d"): 1,
    ("gru", "cross_attention_2d"): 1,  ("gru", "cross_attention_3d"): 1,  ("gru", "cross_attention_4d"): 1,
    ("transformer", "cross_attention_2d"): 1, ("transformer", "cross_attention_3d"): 1, ("transformer", "cross_attention_4d"): 1,
    ("gpt", "cross_attention_2d"): 1, ("gpt", "cross_attention_3d"): 1, ("gpt", "cross_attention_4d"): 1,
    ("transformer_linhead", "cross_attention_2d"): 1, ("transformer_linhead", "cross_attention_3d"): 1,
    ("transformer_linhead", "cross_attention_4d"): 1,
    ("gpt_linhead", "cross_attention_2d"): 1, ("gpt_linhead", "cross_attention_3d"): 1,
    ("gpt_linhead", "cross_attention_4d"): 1,
    ("lstm", "matched_axial_2d"): 1, ("lstm", "matched_axial_3d"): 1, ("lstm", "matched_axial_4d"): 1,
    ("gru", "matched_axial_2d"): 1,  ("gru", "matched_axial_3d"): 1,  ("gru", "matched_axial_4d"): 1,
    ("transformer", "matched_axial_2d"): 1, ("transformer", "matched_axial_3d"): 1, ("transformer", "matched_axial_4d"): 1,
    ("gpt", "matched_axial_2d"): 1, ("gpt", "matched_axial_3d"): 1, ("gpt", "matched_axial_4d"): 1,
    # ACA with SSM backbones: same axial module as above, Mamba-2 / S4 over time.
    ("mamba", "cross_attention_2d"): 1, ("mamba", "cross_attention_3d"): 1, ("mamba", "cross_attention_4d"): 1,
    ("s4", "cross_attention_2d"): 1, ("s4", "cross_attention_3d"): 1, ("s4", "cross_attention_4d"): 1,
    # Mamba-ND scans the dense (S,C,Flow) lattice x time -> small batch.
    ("mamba_nd", "cross_attention_2d"): 1, ("mamba_nd", "cross_attention_3d"): 1, ("mamba_nd", "cross_attention_4d"): 1,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train time series forecasting models"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # s4 = REAL S4 (S4Block/DPLR from external/s4) via src/models/s4.py, so it
        # can run the aggregate (Exp 1) path here too; the per-combo Exp 2 S4 runs
        # still go through the upstream repo (external/s4/train.py). There is
        # no s4d at all: its only implementation lived in the deleted
        # ssm_models.py placeholder, so nothing registers it.
        choices=[
            "lstm", "gru", "transformer", "gpt", "transformer_linhead", "gpt_linhead",
            "xgboost", "lightgbm", "random_forest", "mamba", "mamba_nd", "s4", "s4_bidir"
        ],
        help="Model architecture to train",
    )

    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["onehot", "embeddings", "cross_attention", "film_attention", "identity_cross_attention",
                 "cross_attention_2d", "cross_attention_3d", "cross_attention_4d",
                 "matched_axial_2d", "matched_axial_3d", "matched_axial_4d"],
        help="Encoding variant",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Base configuration file",
    )

    parser.add_argument(
        "--legacy-reproduction", action="store_true",
        help="Use pinned original mixed measures and sum-all duplicate policy.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/runs",
        help="Output directory for checkpoints",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Base directory containing data/ folder",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )

    parser.add_argument(
        "--aggregate", action="store_true",
        help="Test 1: collapse all combos into one summed series.",
    )
    parser.add_argument(
        "--num-layers", type=int, default=None,
        help="Override model layer count (default: base.yaml's 4, all experiments).",
    )
    parser.add_argument(
        "--input-len", type=int, default=None,
        help="Override input window length (months).",
    )
    parser.add_argument(
        "--lag-count", type=int, default=None,
        help="Override number of lag features.",
    )
    parser.add_argument(
        "--train-target-start", default=None,
        help="Inclusive first training-target month; earlier rows remain history.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Ablation switch. base.yaml calls this an open question: A = -exp(A_log), so
    # decaying A_log pulls every head's timescale toward A -> -1 (~7 steps of
    # memory). On 144 training months that shrinkage may act as a useful prior
    # rather than a bug, and the previously reported winner was trained with it.
    parser.add_argument(
        "--no-weight-decay-exempt",
        dest="weight_decay_exempt",
        action="store_false",
        default=None,
        help="Decay every parameter, ignoring the SSM authors' _no_weight_decay "
             "(Mamba) and _optim (S4) marks. Reproduces the pre-2026-07 runs.",
    )

    # Resume is opt-in so a default invocation stays byte-identical to the
    # published protocol; --resume makes a sweep idempotent across a killed box
    # without the caller having to test for last.pth itself.
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from <out-dir>/last.pth when it exists (optimizer, LR "
             "schedule, AMP scaler and early-stopping counter are all restored). "
             "Starts fresh when it does not.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="CKPT",
        help="Continue from an explicit checkpoint path. Errors if absent, "
             "unlike --resume.",
    )
    parser.add_argument(
        "--trackers",
        default=None,
        help="Comma-separated experiment trackers to mirror metrics into "
             "(tensorboard, wandb). Files are written either way; a missing "
             "package is a warning, not an error.",
    )

    parser.add_argument("--exp3", action="store_true", help="Use the Exp 3 head and completion policy")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Seed everything BEFORE any model construction or dataloader instantiation.
    # The returned generator is passed to the training DataLoader for
    # reproducible shuffling. See src/utils/seeding.py.
    rng = set_seed(args.seed)

    print("=" * 80)
    print(f"Training {args.model} with {args.variant} variant (seed={args.seed})")
    print("=" * 80)

    # Load configurations
    print("Loading configurations...")
    cfg = load_config(args.config)

    # Load model-specific config
    model_config_path = f"config/models/{args.model}.yaml"
    if Path(model_config_path).exists():
        cfg = load_config(model_config_path, cfg)
        print(f"  Loaded model config from {model_config_path}")

    # Load variant-specific config
    variant_config_path = f"config/variants/{args.variant}.yaml"
    if Path(variant_config_path).exists():
        cfg = load_config(variant_config_path, cfg)
        print(f"  Loaded variant config from {variant_config_path}")

    if args.exp3:
        if args.aggregate or args.legacy_reproduction or args.variant != "embeddings":
            raise ValueError("--exp3 requires non-aggregate embeddings without legacy reproduction")
        from src.utils.exp3_protocol import apply_exp3_config
        cfg = apply_exp3_config(cfg, args.model)

    # Exp 1 overrides (batch size etc.): the aggregate path must not inherit
    # Exp 2/3's combo-aligned batch of 918 (only ~96 train windows exist).
    # Loaded last so it wins over base/model/variant; CLI flags below still win.
    if args.aggregate and Path("config/exp1.yaml").exists():
        cfg = load_config("config/exp1.yaml", cfg)
        print("  Loaded Exp 1 overrides from config/exp1.yaml")

    if args.legacy_reproduction:
        cfg = apply_legacy_reproduction(cfg, aggregate=args.aggregate)
        require_legacy_output(args.out_dir)

    # Override with command line arguments
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        cfg["training"]["batch_size"] = args.batch_size
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
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        cfg["training"]["epochs"] = args.epochs
        if args.model in ("xgboost", "lightgbm"):
            cfg["model"]["n_estimators"] = args.epochs
    if args.lr is not None:
        if args.lr <= 0:
            raise ValueError("--lr must be positive")
        cfg["training"]["lr"] = args.lr
    if args.weight_decay_exempt is not None:
        cfg["training"]["weight_decay_exempt"] = args.weight_decay_exempt

    # Tree learners subsample rows/features and are therefore stochastic.  The
    # old hard-coded random_state=42 made outputs labelled seed 26/105/115
    # byte-identical fake replicates.  Make the requested experiment seed the
    # actual learner seed and persist it in the resolved model config.
    if args.model in ("xgboost", "lightgbm", "random_forest"):
        cfg["model"]["random_state"] = int(args.seed)

    # Update config with paths
    base_dir = Path(args.data_dir)
    from src.utils.config import resolve_data_paths
    cfg["data"] = resolve_data_paths(cfg["data"], base_dir)
    cfg["checkpointing"]["save_dir"] = args.out_dir
    cfg["logging"]["log_dir"] = str(Path(args.out_dir) / "logs")

    # Set experiment name
    if cfg["logging"]["experiment_name"] is None:
        cfg["logging"]["experiment_name"] = f"{args.model}_{args.variant}"

    # Identity for status.json, so the monitor labels a run from its own record
    # rather than by parsing the directory name.
    cfg["logging"]["run_meta"] = {
        "model_key": args.model,
        "variant_key": args.variant,
        "seed": args.seed,
        "out_dir": str(args.out_dir),
        "aggregate": bool(getattr(args, "aggregate", False)),
    }
    if args.trackers:
        cfg["logging"]["trackers"] = [
            t.strip() for t in args.trackers.split(",") if t.strip()
        ]

    print(f"Configuration loaded")
    print(f"  Model: {args.model}")
    print(f"  Variant: {args.variant}")
    print(f"  Batch size: {cfg['training']['batch_size']}")
    print(f"  Epochs: {cfg['training']['epochs']}")
    print(f"  Learning rate: {cfg['training']['lr']}")
    # Recorded so an ablation run is identifiable from its log alone.
    print(f"  Weight decay exempt: {cfg['training'].get('weight_decay_exempt', True)}")
    print()

    # Create data pipeline
    print("Loading and preprocessing data...")
    # ``splits`` is a top-level config section; get_data_config overlays it on
    # the data block.  Constructing from cfg["data"] silently ignored every
    # user-supplied split boundary because DataConfig defaults masked the bug.
    data_config = DataConfig(**get_data_config(cfg))
    pipeline = TradeDataPipeline(data_config)

    # Load data
    pipeline.load_data()

    # Create splits
    train_df, val_df, test_df = pipeline.create_splits()

    print(f"Data splits created:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print()

    # Detect combo path. Combo variants (cross_attention, film_attention)
    # for LSTM and GRU use a (B, L, G, F) per-group window with group
    # masks; their model classes set requires_combo_loader=True. The
    # dataloader needs to be built in combo mode AND the model factory
    # needs num_combos / features_per_group BEFORE construction.
    # gru/lstm support flat + axial combo variants; transformer/gpt/mamba/s4
    # support only the axial (S4ND-style) variants -- for transformer/gpt because
    # they already self-attend, for mamba/s4 because their flat arms are the
    # 1-D baselines the axial ones are compared against. mamba_nd(_deep) uses
    # its own ND scan and takes only the cross_attention_* rank labels.
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

    # Apply the paper's per-config combo batch size unless the user gave an
    # explicit --batch-size. Done here (not in a YAML) because the value
    # depends on BOTH model and variant, which no single config file knows.
    if combo and not args.batch_size:
        # Every combo arm pins batch 1 = one target month; a variant without a
        # paper entry (identity_cross_attention) gets the same default.
        paper_bs = COMBO_BATCH_SIZES.get((args.model, args.variant), 1)
        cfg["training"]["batch_size"] = paper_bs
        origin = "paper" if (args.model, args.variant) in COMBO_BATCH_SIZES else "default combo"
        print(f"  Combo path: using {origin} batch size {paper_bs} for "
              f"({args.model}, {args.variant}). Override with --batch-size.")

    # Create dataloaders. The test loader is deliberately discarded: training
    # and model selection see only train+val, and the test split is evaluated
    # exactly once, post-training, by scripts/inference.py.
    print(f"Creating dataloaders (combo={combo})...")
    if data_config.batch_by_target_month and args.batch_size is not None:
        expected_batch = 1 if combo else len(pipeline.kept_combos)
        if args.batch_size != expected_batch:
            raise ValueError(
                f"Month-matched training requires --batch-size {expected_batch}; "
                "disable data.batch_by_target_month explicitly for a different protocol"
            )
    train_loader, val_loader, _ = pipeline.get_dataloaders(
        batch_size=cfg["training"]["batch_size"],
        num_workers=4,
        generator=rng,
        worker_init_fn=seed_worker,
        combo=combo,
    )

    run_contract = build_run_contract(
        cfg,
        model=args.model,
        variant=args.variant,
        seed=args.seed,
        aggregate=args.aggregate,
        source_fingerprint=pipeline.source_fingerprint,
        source_domain=pipeline.source_domain,
    )
    cfg["logging"]["run_contract"] = run_contract

    def publish_data_artifacts():
        pipeline.save_artifacts(cfg["checkpointing"]["save_dir"])
        save_run_contract(run_contract, cfg["checkpointing"]["save_dir"])

    # Create model
    print(f"Creating model: {args.model}...")
    # Copy the model config and remove keys that we pass explicitly below.
    # Otherwise create_model(name=..., variant=..., **model_cfg) raises
    # TypeError: got multiple values for keyword argument 'name'/'variant'.
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model_cfg.pop("variant", None)

    # Translate config embedding_dims to the kwarg names BaseModel expects,
    # so the repo-wide standard (state=8, commodity=32, flow=2) is config-driven.
    emb_dims = model_cfg.pop("embedding_dims", None)
    if emb_dims:
        model_cfg.setdefault("state_embed_dim", emb_dims.get("state", 8))
        model_cfg.setdefault("comm_embed_dim", emb_dims.get("commodity", 32))
        model_cfg.setdefault("flow_embed_dim", emb_dims.get("flow", 2))

    # Tree models carry their OWN RNG -- xgboost/lightgbm draw subsample and
    # colsample_bytree from it, sklearn's forest draws its bootstrap samples --
    # and it is seeded by `random_state`, not by set_seed() above. Without this
    # line the run seed never reached them: all three hardcoded random_state=42,
    # so every seed of a tree arm produced a byte-identical model and every
    # reported tree seed-variance was exactly 0. A `random_state` in the model
    # config still wins (setdefault), so a run can still be pinned.
    if args.model in ("xgboost", "lightgbm", "random_forest"):
        model_cfg.setdefault("random_state", args.seed)

    # Get number of categorical features
    num_states = len(pipeline.state2id)
    num_commodities = len(pipeline.comm2id)
    num_flows = len(pipeline.flow2id)

    # Get number of numeric features
    num_numeric_features = len(pipeline.feat_cols)

    # Combo models need to know the group dimension and per-group feature
    # count at construction time.
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

    print(f"Model created with {model.get_num_params():,} parameters")
    print()

    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(cfg)

    # Train
    print("Starting training...")
    print("-" * 80)

    resume_from = args.resume_from or (True if args.resume else None)
    try:
        history = trainer.fit(
            model, train_loader, val_loader, resume_from=resume_from,
            on_ready=publish_data_artifacts,
        )
    except BaseException as exc:
        # Mark the run terminal before the exception leaves the process. Without
        # this the heartbeat is frozen at "running" and the monitor reports a
        # crashed run as merely stalled -- and on Ctrl-C the distinction between
        # "I stopped it" and "it died" is exactly what you want recorded.
        if getattr(trainer, "logger", None) is not None:
            state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            trainer.logger.log_exception(f"run {state}: {type(exc).__name__}: {exc}")
            trainer.logger.close(state=state, summary={"error": f"{type(exc).__name__}: {exc}"})
        raise

    print("-" * 80)
    print("Training completed!")
    print()

    # Print final metrics
    train_loss = history["train_loss"][-1]
    val_loss = history["val_loss"][-1]
    print(f"Final train loss: {train_loss:.6f}")
    print(f"Final val loss: {val_loss:.6f}")
    print()

    if args.model in ("xgboost", "lightgbm", "random_forest"):
        # Tree models save via their native format under the best_xgb prefix
        # (see Trainer._fit_sklearn); best.pth does not exist for them.
        print(f"Best model saved to: {args.out_dir}/best_xgb_*.joblib")
    else:
        print(f"Best model saved to: {args.out_dir}/best.pth")
    print(f"Logs saved to: {cfg['logging']['log_dir']}")

    # Save training loss curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_plots = Path(args.out_dir) / "plots"
        out_plots.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history["train_loss"], label="train loss")
        ax.plot(epochs, history["val_loss"],   label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{args.model}/{args.variant} — training loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_plots / "loss_curve.png", dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to {out_plots}/loss_curve.png")
    except Exception as e:
        print(f"[warn] loss plot failed: {e}")


if __name__ == "__main__":
    main()
