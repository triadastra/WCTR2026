"""Training-time data provenance shared with the vendored S4 runtime.

This module intentionally has no package-relative imports: the vendored S4
process has its own top-level ``src`` package and loads this file directly.
"""

COHORT_PROTOCOL_VERSION = "train-only-cohort-v1"
# Loaded by filename because the vendored runtime owns a different `src` package.
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_s4_shared_benchmark_protocol", Path(__file__).with_name("benchmark_protocol.py"))
_protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_protocol)
BENCHMARK_PROTOCOL_VERSION = _protocol.BENCHMARK_PROTOCOL_VERSION
S4ND_PADDING_VERSION = "fixed-lattice-masked-input-and-layers-v1"


def s4_experiment_identity(experiment):
    bidirectional = experiment.endswith("_bidir")
    name = experiment.removesuffix("_bidir")
    model, variant = name.split("_", 1)
    if bidirectional:
        if model != "s4" or variant not in {"embeddings", "onehot"}:
            raise ValueError(f"Unsupported bidirectional S4 experiment: {experiment}")
        model = "s4_bidir"
    return model, variant


def s4_run_contract(config, dataset):
    """Bind S4 weights to resolved architecture, data, and training settings.

    Keep this module independent of either runtime's package namespace.
    The loss revision invalidates earlier padded-denominator checkpoints.
    """
    from dataclasses import asdict
    import copy

    metadata = checkpoint_cohort_metadata(dataset)
    if metadata is None:
        raise ValueError("S4 benchmark contract requires a trade dataset")
    name = config["dataset"]["_name_"]
    if name == "s4_onehot_u":
        name = "s4_onehot"
    model, variant = name.split("_", 1)
    if model == "s4" and config["model"]["layer"].get("bidirectional", False):
        model = "s4_bidir"
    data = asdict(dataset.pipe.config)
    for key in ("imports_dir", "exports_dir", "aggregated_csv"):
        data.pop(key, None)
    training = {key: copy.deepcopy(config.get(key, {})) for key in (
        "optimizer", "scheduler", "loader", "loader_train", "trainer", "train", "callbacks"
    )}
    # These are runtime locations/actions, not the optimization recipe.
    for key in ("ckpt", "seed", "test"):
        training["train"].pop(key, None)
    for key in ("dirpath", "filename"):
        training.get("callbacks", {}).get("model_checkpoint", {}).pop(key, None)
    training["loss_reduction"] = "valid-target-mean-v2" if model == "s4nd" else "valid-target-mean-v1"
    model_config = {key: copy.deepcopy(config.get(key)) for key in (
        "model", "encoder", "decoder", "task"
    )}
    if model == "s4nd":
        # Code-level behavior must invalidate resume, evaluation and skip gates
        # even if a checkpoint's saved YAML predates this correction.
        model_config["grid_padding"] = S4ND_PADDING_VERSION
    training["allow_tf32"] = False
    return {
        "run_contract_version": 1,
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "identity": {"model": model, "variant": variant,
                     "seed": int(config["train"]["seed"]), "aggregate": False},
        "model_config": model_config,
        "data_config": data,
        "training_config": training,
        "selection": {"monitor": config["train"]["monitor"],
                      "mode": config["train"]["mode"]},
        "source_fingerprint": metadata["source_fingerprint"],
        "source_domain": metadata["source_domain"],
    }


def checkpoint_cohort_metadata(dataset):
    """Build exact source/cohort metadata for a configured trade dataset."""
    pipe = getattr(dataset, "pipe", None)
    kept_combos = getattr(pipe, "kept_combos", None)
    if pipe is None or kept_combos is None:
        return None
    source_fingerprint = getattr(pipe, "source_fingerprint", None)
    source_domain = getattr(pipe, "source_domain", None)
    if not source_fingerprint or not isinstance(source_domain, dict):
        raise RuntimeError(
            "trade dataset has no verified source fingerprint/domain"
        )
    return {
        "protocol_version": COHORT_PROTOCOL_VERSION,
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "cohort_from_train_only": bool(
            getattr(pipe.config, "cohort_from_train_only", True)
        ),
        "num_combos": len(kept_combos),
        "combos": [list(combo) for combo in kept_combos],
        "source_fingerprint": str(source_fingerprint),
        "source_domain": source_domain,
        "data_protocol": {
            "input_len": int(pipe.config.input_len),
            "lag_count": int(pipe.config.lag_count),
            "train_target_start": (
                None
                if pipe.config.train_target_start in (None, "")
                else str(pipe.config.train_target_start)[:10]
            ),
        },
    }


def validate_checkpoint_cohort(checkpoint, dataset):
    """Return verified training metadata or reject an unverifiable checkpoint."""
    recorded = checkpoint.get("cohort")
    if not isinstance(recorded, dict):
        raise RuntimeError(
            "checkpoint has no training-time cohort provenance; retrain it "
            "under the corrected protocol before evaluation"
        )
    if recorded.get("protocol_version") != COHORT_PROTOCOL_VERSION:
        raise RuntimeError("checkpoint uses an unknown or legacy cohort protocol")
    if recorded.get("benchmark_protocol_version") != BENCHMARK_PROTOCOL_VERSION:
        raise RuntimeError("checkpoint predates the current benchmark protocol")
    if recorded.get("cohort_from_train_only") is not True:
        raise RuntimeError("checkpoint was not trained on a train-only cohort")

    current = checkpoint_cohort_metadata(dataset)
    if current is None:
        raise RuntimeError("evaluation dataset does not expose its kept cohort")
    expected = recorded.get("combos")
    if not isinstance(expected, list) or current["combos"] != expected:
        raise RuntimeError(
            "evaluation cohort does not match the cohort stored in the checkpoint"
        )
    if recorded.get("num_combos") != len(expected):
        raise RuntimeError("checkpoint cohort count does not match its combo list")
    for field in (
        "source_fingerprint", "source_domain", "data_protocol",
    ):
        if recorded.get(field) != current[field]:
            raise RuntimeError(
                f"evaluation {field} does not match the training checkpoint"
            )
    return recorded
