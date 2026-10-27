"""Evaluate validation-selected S4/S4ND checkpoints with shared metrics.

Run from the repository root: python scripts/s4_eval.py --seed 26
The saved training configuration and contract are authoritative. Source paths
may be relocated through DATA_DIR. Missing/mismatched checkpoints fail closed.
"""
import json
import os
import sys
from pathlib import Path

# Running `python /root/s4_eval.py` puts /root (not cwd) on sys.path[0];
# prepend the external/s4 working dir so the local `src` package resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external/s4"))

import numpy as np
import torch
from omegaconf import OmegaConf

import src.utils.train  # noqa: F401  (registers process_config)
from train import SequenceLightningModule

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "utils"))
from s4_checkpoint_provenance import validate_checkpoint_cohort, s4_run_contract, BENCHMARK_PROTOCOL_VERSION
from s4_benchmark import select_valid_targets
from s4_completion import read_s4_completion


def _shared_module(name, relative_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_shared_metrics = _shared_module("benchmark_metrics", "src/inference/metrics.py")
_scaled_metrics = _shared_module("benchmark_scaled_metrics", "src/inference/scaled_metrics.py")


def cohort_provenance(num_combos, train_only):
    # Kept local because this script deliberately imports external/s4/src as
    # ``src`` when executed from the upstream S4 working directory.
    return {
        "protocol_version": "train-only-cohort-v1",
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "cohort_from_train_only": bool(train_only),
        "num_combos": int(num_combos),
    }


RUNS = Path(__file__).resolve().parents[1] / "outputs" / "full"  # repo/outputs/full (was /root/full/...)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# experiment -> run dir under outputs/full
# Keys = experiments eligible for eval when no argv filter is given. Values are
# legacy seed-42 dir names (no longer used — main() discovers all seed dirs by
# pattern). dss_* = the REAL Diagonal State Space (complex-diagonal, disc=dss).
EXPERIMENTS = {
    "s4_embeddings":  "s4_s4_embeddings_seed42",
    "s4_onehot":      "s4_s4_onehot_seed42",
    "s4_embeddings_bidir": "s4_s4_embeddings_bidir_seed42",
    "s4_onehot_bidir": "s4_s4_onehot_bidir_seed42",
    "s4nd_2d":        "s4_s4nd_2d_seed42",
    "s4nd_2d_state":  "s4_s4nd_2d_state_seed42",
    "s4nd_3d":        "s4_s4nd_3d_seed42",
    "s4nd_4d":        "s4_s4nd_4d_seed42",
}


def to_device(batch):
    x, y, *z = batch
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    if z:
        z = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in z[0].items()}]
    return (x, y, *z)


def _percombo_plots(predictions_csv, plots_dir, title, top_n=25):
    """Per-combo predicted-vs-actual over the full test horizon (separate plot per
    combo) — same convention as the github models; no sum-over-combos aggregate.
    Inline matplotlib to avoid the github-vs-s4 `src` package name clash."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    plots_dir = Path(plots_dir)
    (plots_dir / "combos").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(predictions_csv, parse_dates=["Time"])
    kc = ["State", "Commodity", "Import/Export"]
    ranking = df.groupby(kc)["true_value"].mean().sort_values(ascending=False)
    keys = ranking.index if (top_n is None or top_n < 0) else ranking.head(top_n).index
    for i, key in enumerate(keys):
        sub = df[(df["State"] == key[0]) & (df["Commodity"] == key[1])
                 & (df["Import/Export"] == key[2])].sort_values("Time")
        safe = "".join(c if c.isalnum() or c in "_-" else "_"
                       for c in "_".join(str(k) for k in key))[:80]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(sub["Time"], sub["true_value"], marker="o", label="actual", color="tab:blue")
        ax.plot(sub["Time"], sub["pred_value"], marker="x", ls="--", label="predicted", color="tab:red")
        ax.set_title(f"{title} — {key[0]} / {key[1]} / {key[2]} (test horizon)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Value ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plots_dir / "combos" / f"{i:03d}_{safe}_value.png", dpi=150)
        plt.close(fig)


def evaluate(exp, run_dir, seed=42):
    ckpt = RUNS / run_dir / "hydra/checkpoints/val/loss.ckpt"
    # A failed evaluation must not leave a stale completion marker behind.
    (RUNS / run_dir / "metrics.json").unlink(missing_ok=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing validation-selected checkpoint: {ckpt}")
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint.get("benchmark_config"), dict):
        raise RuntimeError("Checkpoint has no resolved training configuration")
    config = OmegaConf.create(checkpoint["benchmark_config"])
    base_experiment = exp.removesuffix("_bidir")
    expected_dataset = "s4_onehot_u" if base_experiment == "s4_onehot" else base_experiment
    if config.dataset._name_ != expected_dataset or int(config.train.seed) != seed:
        raise RuntimeError("Checkpoint experiment/seed does not match run directory")
    if base_experiment in {"s4_embeddings", "s4_onehot"}:
        if bool(config.model.layer.get("bidirectional", False)) != exp.endswith("_bidir"):
            raise RuntimeError("Checkpoint S4 direction does not match run directory")
    # Only the source location may move between training and evaluation hosts.
    if os.environ.get("DATA_DIR"):
        config.dataset.data_dir = os.path.abspath(os.environ["DATA_DIR"])
    config.wandb = None
    model = SequenceLightningModule(config)
    training_cohort = validate_checkpoint_cohort(checkpoint, model.dataset)
    contract = s4_run_contract(model._benchmark_config, model.dataset)
    if checkpoint.get("run_contract") != contract:
        raise RuntimeError("Checkpoint lacks the current S4 training/loss contract; retrain")
    training_completion = read_s4_completion(ckpt, contract)
    state = checkpoint["state_dict"]
    result = model.load_state_dict(state, strict=False)
    # Surface any key mismatch -- silent strict=False could otherwise hide a
    # checkpoint that doesn't match the rebuilt model (wrong d_input, encoder,
    # etc.). S4 keeps a few non-persistent kernel buffers, so a *small* number
    # of missing keys is normal; unexpected keys or many missing ones are not.
    miss = [k for k in result.missing_keys if not k.endswith(("kernel.L", "kernel.w"))]
    if miss or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint parameter mismatch: missing={miss}, unexpected={result.unexpected_keys}")
    model.eval().to(DEVICE)

    loader = model.dataset.test_dataloader(**config.loader)
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    preds, targs = [], []
    model._initialize_state()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = to_device(batch)
            model._process_state(batch, i, train=False)
            x, y, _ = model.forward(batch)
            extra = batch[2] if len(batch) > 2 and isinstance(batch[2], dict) else {}
            x, y = select_valid_targets(x, y, model.dataset, extra)
            preds.append(x.reshape(-1, 2).float().cpu().numpy())
            targs.append(y.reshape(-1, 2).float().cpu().numpy())

    pred = np.concatenate(preds).astype("float64")
    targ = np.concatenate(targs).astype("float64")
    import pandas as pd
    keys = model.dataset.test_keys
    times = model.dataset.test_times
    if len(keys) != len(pred) or len(times) != len(pred):
        raise RuntimeError("Prediction keys/times do not align with kept targets")
    dn = model.dataset._denorm
    if any(k not in dn for k in keys):
        raise RuntimeError("Missing train-fitted inverse scale for evaluated series")
    scales = np.asarray([dn[k] for k in keys], dtype=float)
    offsets, ranges = scales[:, [0, 2]], scales[:, [1, 3]]
    pred_real, targ_real = pred * ranges + offsets, targ * ranges + offsets
    m = _shared_metrics.compute_evaluation_metrics(pred, targ, pred_real, targ_real)
    m["num_params"] = sum(p.numel() for p in model.parameters())
    _scaled_metrics.add_mase_rmsse(m, model.dataset.pipe, pred, targ, keys)
    frame = pd.DataFrame(keys, columns=["State", "Commodity", "Import/Export"])
    frame["Time"] = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in times]
    for values, prefix in ((pred, "pred"), (targ, "true")):
        for i, name in enumerate(("value", "weight")):
            frame[f"{prefix}_{name}"] = values[:, i]
    for values, prefix in ((pred_real, "pred"), (targ_real, "true")):
        for i, name in enumerate(("value", "weight")):
            frame[f"{prefix}_{name}_real"] = values[:, i]
    outdir = RUNS / run_dir
    frame.to_csv(outdir / "predictions_norm.csv", index=False)
    physical = frame.copy()
    for prefix in ("pred", "true"):
        for name in ("value", "weight"):
            physical[f"{prefix}_{name}"] = physical[f"{prefix}_{name}_real"]
    physical.to_csv(outdir / "predictions.csv", index=False)
    fam, variant = contract["identity"]["model"], contract["identity"]["variant"]
    measurement_module = _shared_module("report_model_measurements", "src/utils/model_measurements.py")
    out = {
        "model_measurements": measurement_module.model_measurements(model, fam),
        "model": fam, "variant": variant, "aggregate": False,
        "mode": "one_step_normalized", "seed": seed,
        "prediction_units": "normalized", "scale_dependent_metric_units": "normalized",
        "percentage_metric_units": "real", "n_samples": len(pred),
        "predictions_file": "predictions_norm.csv", "checkpoint": str(ckpt),
        "source_fingerprint": training_cohort["source_fingerprint"],
        "source_domain": training_cohort["source_domain"], "run_contract": contract,
        "training_completion": training_completion,
        "architecture": {
            "class": str(config.model.layer.get("_name_", "")),
            "n_layers": int(config.model.get("n_layers", 0)),
            "d_state": int(config.model.layer.get("d_state", 0)),
            "bidirectional": bool(config.model.layer.get("bidirectional", False)),
            "dim": int(config.model.layer.get("dim", 1)),
        },
        "cohort": cohort_provenance(training_cohort["num_combos"], True),
        "data_protocol": training_cohort["data_protocol"], "metrics": m,
    }
    (outdir / "metrics.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    try:
        _percombo_plots(outdir / "predictions.csv", outdir / "plots", f"{fam}/{variant}")
    except Exception as exc:
        print(f"[warn] Optional plots failed: {exc}")
    print(f"[ok] {exp}: n={pred.shape[0]}  norm_mse={m['norm_mse_combined']:.5f}  "
          f"mae={m['mae']:.4f}  r2={m['r2']:.4f}")
    return out


def main():
    import argparse
    import re
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", nargs="*", metavar="EXPERIMENT")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--runs-dir", type=Path, default=RUNS)
    args = ap.parse_args()
    unknown = set(args.experiments) - set(EXPERIMENTS)
    if unknown:
        ap.error(f"Unknown S4 experiments: {sorted(unknown)}")
    global_runs = args.runs_dir.resolve()
    globals()["RUNS"] = global_runs
    only = set(args.experiments) or set(EXPERIMENTS)
    pat = re.compile(r"^s4_(?P<exp>.+)_seed(?P<seed>\d+)$")
    selected = []
    for path in sorted(global_runs.iterdir()) if global_runs.is_dir() else []:
        match = pat.fullmatch(path.name)
        if match and match["exp"] in only and (args.seed is None or int(match["seed"]) == args.seed):
            selected.append((path.name, match["exp"], int(match["seed"])))
    missing = only - {exp for _, exp, _ in selected}
    failures = len(missing)
    if missing:
        print(f"[FAIL] No runs found for: {sorted(missing)}")
    for directory, exp, seed in selected:
        try:
            evaluate(exp, directory, seed)
        except Exception as exc:
            import traceback
            print(f"[FAIL] {directory}: {exc}")
            traceback.print_exc()
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
