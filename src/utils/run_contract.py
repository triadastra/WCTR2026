"""Persist and validate the semantic contract of a trained run.

The best checkpoint alone is not enough provenance: most sequence models can
load the same weights under a different input-window length, seed, or split
calendar without a shape error.  This module records the resolved settings that
define an experiment and makes inference reject a checkpoint/config mismatch
instead of silently relabelling the result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .config import get_data_config, load_config
from .provenance import BENCHMARK_PROTOCOL_VERSION


RUN_CONTRACT_VERSION = 1
_DATA_PATH_KEYS = {"imports_dir", "exports_dir", "aggregated_csv"}
_NATIVE_TREE_MODELS = {"xgboost", "lightgbm", "random_forest"}


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def semantic_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolved data/split config, excluding machine-specific source paths."""
    resolved = get_data_config(dict(config))
    # Omission and explicit null both mean the DataConfig default. The
    # canonical Exp1 contract spells this out; CLI runs must serialize the
    # same semantics even when no --train-target-start override was supplied.
    resolved.setdefault("train_target_start", None)
    return _json_value({
        key: value for key, value in resolved.items() if key not in _DATA_PATH_KEYS
    })


def _model_config(config: Mapping[str, Any], model: str) -> dict[str, Any]:
    resolved = dict(config.get("model", {}) or {})
    if model in {"s4", "s4_bidir"}:
        # Bind even hand-built configs to the actual backend. Use the same
        # normalization when saving and validating; never upgrade saved contracts.
        resolved.setdefault("implementation", "authors-modular-s4-v1")
    return _json_value(resolved)


def build_run_contract(
    config: Mapping[str, Any],
    *,
    model: str,
    variant: str,
    seed: int,
    aggregate: bool,
    source_fingerprint: str,
    source_domain: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable, semantic description of one training run."""
    if not source_fingerprint:
        raise ValueError("run contract requires a non-empty source fingerprint")
    if not isinstance(source_domain, Mapping) or not source_domain:
        raise ValueError("run contract requires a non-empty source domain")
    checkpointing = config.get("checkpointing", {}) or {}
    return {
        # Both aggregate and per-combo training call the same seed/runtime
        # setup. Do not accept old aggregate checkpoints whose TF32 arithmetic
        # was inherited from the runtime default under the corrected policy.
        "numerics": {"allow_tf32": False},
        "run_contract_version": RUN_CONTRACT_VERSION,
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "identity": {
            "model": str(model),
            "variant": str(variant),
            "seed": int(seed),
            "aggregate": bool(aggregate),
        },
        "model_config": _model_config(config, model),
        "data_config": semantic_data_config(config),
        "training_config": _json_value(config.get("training", {}) or {}),
        "selection": {
            "monitor": checkpointing.get("monitor", "val_loss"),
            "mode": checkpointing.get("mode", "min"),
        },
        "source_fingerprint": str(source_fingerprint),
        "source_domain": _json_value(source_domain),
    }


def expected_exp1_run_contract(
    repo: str | Path,
    run: Any,
    protocol: Mapping[str, Any],
    source_domain: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact contract for one canonical Experiment 1 run.

    Both the resume/skip gate and the paper-results collector call this helper.
    Keeping one reconstruction path prevents the launcher from accepting an
    off-protocol artifact that the collector rejects only after all expensive
    training has finished.
    """
    repo = Path(repo)
    config = load_config(str(repo / "config" / "base.yaml"))
    model_config = repo / "config" / "models" / f"{run.model}.yaml"
    if model_config.is_file():
        config = load_config(str(model_config), config)
    variant_config = repo / "config" / "variants" / f"{run.variant}.yaml"
    if variant_config.is_file():
        config = load_config(str(variant_config), config)
    config = load_config(str(repo / "config" / "exp1.yaml"), config)
    config["data"]["aggregate"] = True
    config["data"]["input_len"] = protocol["input_len"]
    config["data"]["lag_count"] = protocol["lag_count"]
    config["data"]["train_target_start"] = protocol["train_target_start"]
    config["training"]["batch_size"] = protocol["batch_size"]
    if run.kind == "tree":
        config["model"]["random_state"] = run.seed
    elif run.kind == "baseline":
        config["model"] = {
            "name": run.model,
            "variant": "none",
            "window": (
                protocol["moving_average_window"]
                if run.model == "moving_average" else None
            ),
        }
    return build_run_contract(
        config,
        model=run.model,
        variant=run.variant,
        seed=run.seed,
        aggregate=True,
        source_fingerprint=str(protocol["source_fingerprint"]),
        source_domain=source_domain,
    )


def save_run_contract(contract: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write the contract beside the other training artifacts."""
    path = Path(output_dir) / "artifacts" / "run_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(_json_value(contract), indent=2, allow_nan=False))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_run_contract(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load the contract belonging to ``checkpoint_path`` or fail closed."""
    path = Path(checkpoint_path).expanduser().resolve().parent / "artifacts" / "run_contract.json"
    if not path.is_file():
        raise ValueError(
            f"Checkpoint is missing its run contract at {path}; retraining under "
            f"benchmark protocol {BENCHMARK_PROTOCOL_VERSION} is required"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint run contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint run contract at {path}: expected an object")
    if payload.get("run_contract_version") != RUN_CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported run contract at {path}: expected version "
            f"{RUN_CONTRACT_VERSION}, got {payload.get('run_contract_version')!r}"
        )
    if payload.get("benchmark_protocol_version") != BENCHMARK_PROTOCOL_VERSION:
        raise ValueError(
            f"Checkpoint run contract at {path} predates benchmark protocol "
            f"{BENCHMARK_PROTOCOL_VERSION}; retraining is required"
        )
    payload["_artifact_path"] = str(path)
    return payload


def validate_run_contract(
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    model: str,
    variant: str,
    seed: int,
    aggregate: bool,
    source_fingerprint: str,
    source_domain: Mapping[str, Any],
) -> None:
    """Reject inference that does not reconstruct the trained experiment."""
    expected_identity = {
        "model": str(model),
        "variant": str(variant),
        "seed": int(seed),
        "aggregate": bool(aggregate),
    }
    mismatches = []
    if contract.get("identity") != expected_identity:
        mismatches.append("identity")
    # A torch checkpoint is instantiated from the current architecture config,
    # so every architecture field must match. Native tree artifacts deserialize
    # the fitted estimators themselves; constructor hyperparameters such as
    # n_estimators are training-only and may legitimately differ at inference
    # (notably for --epochs smoke runs). Their saved metadata independently
    # validates the feature/categorical dimensions after loading.
    if (
        model not in _NATIVE_TREE_MODELS
        and contract.get("model_config") != _model_config(config, model)
    ):
        mismatches.append("model_config")
    if contract.get("data_config") != semantic_data_config(config):
        mismatches.append("data_config")
    # Training settings and checkpoint-selection policy are provenance, not
    # inference inputs. Enforcing current epochs/lr/optimizer/patience here made
    # correctly trained artifacts impossible to evaluate after harmless config
    # edits and made every --epochs smoke checkpoint fail. The strict paper
    # collector reconstructs and compares the complete contract before it can
    # publish any result.
    if not isinstance(contract.get("training_config"), Mapping):
        mismatches.append("training_config")
    if not isinstance(contract.get("selection"), Mapping):
        mismatches.append("selection")
    if contract.get("source_fingerprint") != str(source_fingerprint):
        mismatches.append("source_fingerprint")
    if contract.get("source_domain") != _json_value(source_domain):
        mismatches.append("source_domain")
    if mismatches:
        location = contract.get("_artifact_path", "checkpoint run contract")
        raise ValueError(
            f"Inference does not match the trained run contract at {location} "
            f"({', '.join(mismatches)}); use the training model, variant, seed, "
            "architecture, data, and split configuration"
        )


def validate_embedded_contract(
    embedded: Any,
    artifact: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
) -> None:
    """Bind the checkpoint bytes logically to the adjacent run artifact."""
    if not isinstance(embedded, Mapping):
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no embedded run contract; "
            f"retraining under benchmark protocol {BENCHMARK_PROTOCOL_VERSION} is required"
        )
    expected = {key: value for key, value in artifact.items() if not str(key).startswith("_")}
    if _json_value(embedded) != _json_value(expected):
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not belong to its adjacent run_contract.json"
        )
