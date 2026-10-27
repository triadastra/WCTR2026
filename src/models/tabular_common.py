"""Shared tabular feature extraction for tree-based forecasting models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from ..utils.run_artifacts import sync_directory, write_checkpoint_manifest


SUPPORTED_TABULAR_VARIANTS = {"embeddings", "onehot"}
TARGET_COLS = {"Value", "Weight"}
KEY_COLS = ["State", "Commodity", "Import/Export"]

# The window-end (month t-1) actuals are the single strongest predictor for a
# near-random-walk monthly series. The neural models see them (raw Value/Weight
# channels at the last timestep); the trees MUST too, or the comparison is
# unfair. We keep them as features under these names (they are the LAST OBSERVED
# value at t-1, NOT the target at t — so no leakage).
LAST_OBS_RENAME = {"Value": "value_last", "Weight": "weight_last"}
ID_FEATURE_NAMES = ["state_id", "comm_id", "flow_id"]


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def save_tree_bundle(
    prefix: str | Path,
    value_model: Any,
    weight_model: Any,
    metadata: dict[str, Any],
) -> None:
    """Atomically publish two joblib estimators and hash-bind their metadata.

    The metadata file is the completion marker and is replaced last.  A crash
    or a swapped ``*_value.joblib``/``*_weight.joblib`` therefore fails closed
    at load time instead of silently evaluating different estimator bytes.
    """
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    final = {
        "value": Path(f"{prefix}_value.joblib"),
        "weight": Path(f"{prefix}_weight.joblib"),
    }
    temporary = {
        name: path.with_name(f".{path.name}.tmp-{os.getpid()}")
        for name, path in final.items()
    }
    metadata_path = Path(f"{prefix}_metadata.json")
    metadata_temp = metadata_path.with_name(
        f".{metadata_path.name}.tmp-{os.getpid()}"
    )
    try:
        dumped_value = joblib.dump(value_model, temporary["value"])
        dumped_weight = joblib.dump(weight_model, temporary["weight"])
        if dumped_value != [str(temporary["value"])] or dumped_weight != [str(temporary["weight"])]:
            raise RuntimeError("tree checkpoint unexpectedly produced multiple files")
        payload = dict(metadata)
        payload["artifact_schema_version"] = 1
        payload["artifact_hashes"] = {
            name: _file_identity(path) for name, path in temporary.items()
        }
        metadata_temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        for path in (*temporary.values(), metadata_temp):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        for name in ("value", "weight"):
            os.replace(temporary[name], final[name])
        os.replace(metadata_temp, metadata_path)
        sync_directory(prefix.parent)
        write_checkpoint_manifest(prefix.parent, {
            "value_estimator": final["value"].name,
            "weight_estimator": final["weight"].name,
            "metadata": metadata_path.name,
        }, kind="tree", run_contract=metadata.get("run_contract"))
    finally:
        for path in (*temporary.values(), metadata_temp):
            path.unlink(missing_ok=True)


def load_tree_bundle(prefix: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    """Verify and load a hash-bound native tree checkpoint bundle."""
    prefix = Path(prefix)
    metadata_path = Path(f"{prefix}_metadata.json")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read tree checkpoint metadata {metadata_path}: {exc}") from exc
    if metadata.get("artifact_schema_version") != 1:
        raise ValueError("Tree checkpoint is legacy or has no artifact hash contract")
    expected_hashes = metadata.get("artifact_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Tree checkpoint metadata has no artifact hashes")
    models = {}
    for name in ("value", "weight"):
        path = Path(f"{prefix}_{name}.joblib")
        expected = expected_hashes.get(name)
        if not isinstance(expected, dict) or _file_identity(path) != expected:
            raise ValueError(
                f"Tree checkpoint component failed integrity validation: {path}"
            )
        models[name] = joblib.load(path)
    return models["value"], models["weight"], metadata


def to_feature_frame(
    X: np.ndarray,
    feature_names: list[str],
    variant: str,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> pd.DataFrame:
    """Wrap the feature matrix in a DataFrame, marking the categorical ID columns
    as pandas `category` dtype (embeddings variant only) so XGBoost
    (enable_categorical) and LightGBM (auto-detect) split on them NATIVELY
    instead of treating the integer IDs as ordinal/continuous. Categories are
    pinned to the full id range so train/val/test share identical codes."""
    df = pd.DataFrame(X, columns=feature_names)
    if variant == "embeddings":
        ranges = {"state_id": num_states, "comm_id": num_commodities, "flow_id": num_flows}
        for col, n in ranges.items():
            if col in df.columns:
                codes = df[col].round().astype("int64")
                df[col] = pd.Categorical(codes, categories=range(n))
    return df


def validate_tabular_variant(model_name: str, variant: str) -> None:
    if variant not in SUPPORTED_TABULAR_VARIANTS:
        raise ValueError(
            f"{model_name} only supports 'embeddings' (integer IDs) and "
            "'onehot' variants in the tabular tree path."
        )


def category_feature_names(
    variant: str,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> list[str]:
    if variant == "onehot":
        return (
            [f"state_{i}" for i in range(num_states)]
            + [f"commodity_{i}" for i in range(num_commodities)]
            + [f"flow_{i}" for i in range(num_flows)]
        )
    return ["state_id", "comm_id", "flow_id"]


def fallback_feature_names(
    variant: str,
    num_numeric_features: int,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> list[str]:
    # num_numeric_features = value_last + weight_last + lag_count value-lags +
    # lag_count weight-lags + sin + cos  =>  lag_count = (n - 4) / 2.
    lag_count = max((num_numeric_features - 4) // 2, 0)
    numeric_names = (
        ["value_last", "weight_last"]
        + [f"value_lag_{k}" for k in range(1, lag_count + 1)]
        + [f"weight_lag_{k}" for k in range(1, lag_count + 1)]
        + ["sin_month", "cos_month"]
    )
    return numeric_names + category_feature_names(
        variant, num_states, num_commodities, num_flows
    )


def category_features(
    variant: str,
    state_ids: torch.Tensor,
    comm_ids: torch.Tensor,
    flow_ids: torch.Tensor,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> np.ndarray:
    if variant == "onehot":
        parts = [
            torch.nn.functional.one_hot(state_ids, num_states),
            torch.nn.functional.one_hot(comm_ids, num_commodities),
            torch.nn.functional.one_hot(flow_ids, num_flows),
        ]
        return torch.cat(parts, dim=-1).cpu().numpy().astype(np.float32)

    return (
        torch.stack([state_ids, comm_ids, flow_ids], dim=-1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def has_trade_dataset_contract(dataset: Any) -> bool:
    return all(
        hasattr(dataset, attr)
        for attr in (
            "df",
            "feat_cols",
            "state2id",
            "comm2id",
            "flow2id",
            "target_time_start",
            "target_time_end",
        )
    )


def extract_tabular_xy(
    loader: Any,
    *,
    variant: str,
    num_numeric_features: int,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = getattr(loader, "dataset", None)
    if has_trade_dataset_contract(dataset):
        return extract_from_trade_dataset(
            dataset,
            variant=variant,
            num_states=num_states,
            num_commodities=num_commodities,
            num_flows=num_flows,
        )

    return extract_from_batches(
        loader,
        variant=variant,
        num_numeric_features=num_numeric_features,
        num_states=num_states,
        num_commodities=num_commodities,
        num_flows=num_flows,
    )


def extract_from_trade_dataset(
    dataset: Any,
    *,
    variant: str,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if getattr(dataset, "tabular_feature_mode", "last_row") == "window":
        return extract_window_samples(
            dataset, variant=variant, num_states=num_states,
            num_commodities=num_commodities, num_flows=num_flows,
        )
    # Keep ALL per-step features incl. Value/Weight: on the feature row (t-1)
    # these are the LAST-OBSERVED actuals, not the target (which is row t). The
    # neural models see them; dropping them here was the unfair information gap.
    feat_cols_used = list(dataset.feat_cols)
    numeric_names = [LAST_OBS_RENAME.get(c, c) for c in feat_cols_used]
    feature_names = numeric_names + category_feature_names(
        variant, num_states, num_commodities, num_flows
    )

    X_list, y_list = [], []
    for (state, comm, flow), group in dataset.df.groupby(KEY_COLS, sort=False):
        if (
            state not in dataset.state2id
            or comm not in dataset.comm2id
            or flow not in dataset.flow2id
        ):
            continue

        group = group.sort_values("Time").reset_index(drop=True)
        if len(group) < 2:
            continue

        feat_prev = group.iloc[:-1].reset_index(drop=True)
        targ_next = group.iloc[1:].reset_index(drop=True)

        mask = np.ones(len(targ_next), dtype=bool)
        if dataset.target_time_start is not None:
            mask &= targ_next["Time"].to_numpy() >= np.datetime64(
                dataset.target_time_start
            )
        # Cross-family target alignment. Set on the TRAIN dataset only, and
        # only when DataConfig.tabular_align_targets is on: the earliest
        # target the windowed arms can emit. Without it a tree row needs just
        # 2 months of history and starts ~3 years earlier than the neural
        # arms it is benchmarked against. Features are untouched -- this
        # clamps which targets are fit, not what the model sees per row.
        align_start = getattr(dataset, "tabular_target_time_start", None)
        if align_start is not None:
            mask &= targ_next["Time"].to_numpy() >= np.datetime64(align_start)
        if dataset.target_time_end is not None:
            mask &= targ_next["Time"].to_numpy() <= np.datetime64(
                dataset.target_time_end
            )
        if not mask.any():
            continue

        x_numeric = feat_prev.loc[mask, feat_cols_used].to_numpy(np.float32)
        n = x_numeric.shape[0]
        state_ids = torch.full((n,), dataset.state2id[state], dtype=torch.long)
        comm_ids = torch.full((n,), dataset.comm2id[comm], dtype=torch.long)
        flow_ids = torch.full((n,), dataset.flow2id[flow], dtype=torch.long)
        x_cat = category_features(
            variant,
            state_ids,
            comm_ids,
            flow_ids,
            num_states,
            num_commodities,
            num_flows,
        )

        y = targ_next.loc[mask, ["Value", "Weight"]].to_numpy(np.float32)
        X_list.append(np.concatenate([x_numeric, x_cat], axis=1))
        y_list.append(y)

    if not X_list:
        raise ValueError("No tabular supervised rows could be built")

    return np.vstack(X_list), np.vstack(y_list), feature_names


def extract_window_samples(
    dataset: Any, *, variant: str, num_states: int,
    num_commodities: int, num_flows: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Use the exact neural samples, including targets omitted by drop_last.

    Flatten oldest-to-newest with feature order unchanged. Category columns
    retain their standard names so boosters still recognize native categories.
    Reading samples directly makes extraction independent of loader shuffling.
    """
    samples = dataset.samples
    if not samples:
        raise ValueError("No windowed supervised rows could be built")
    width = dataset.input_len * len(dataset.feat_cols)
    numeric = np.empty((len(samples), width), dtype=np.float32)
    targets = np.empty((len(samples), 2), dtype=np.float32)
    for i, sample in enumerate(samples):
        numeric[i] = sample["x_numeric"].detach().cpu().numpy().reshape(-1)
        targets[i] = [float(sample["target_value"]), float(sample["target_weight"])]
    categorical = category_features(
        variant,
        torch.tensor([s["state_id"] for s in samples], dtype=torch.long),
        torch.tensor([s["comm_id"] for s in samples], dtype=torch.long),
        torch.tensor([s["flow_id"] for s in samples], dtype=torch.long),
        num_states, num_commodities, num_flows,
    )
    names = [
        f"history_{dataset.input_len - step}__{feature}"
        for step in range(dataset.input_len) for feature in dataset.feat_cols
    ] + category_feature_names(variant, num_states, num_commodities, num_flows)
    return np.concatenate([numeric, categorical], axis=1), targets, names


def extract_from_batches(
    loader: Any,
    *,
    variant: str,
    num_numeric_features: int,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_list, y_list = [], []
    with torch.no_grad():
        for batch in loader:
            x_numeric = batch["x_numeric"]
            state_ids = batch["state_ids"]
            comm_ids = batch["comm_ids"]
            flow_ids = batch["flow_ids"]

            # Include ALL channels at the last timestep (t-1), incl. Value/Weight
            # (index 0,1) — the last-observed actuals the neural models also see.
            x_last = x_numeric[:, -1, :].cpu().numpy().astype(np.float32)
            x_cat = category_features(
                variant,
                state_ids,
                comm_ids,
                flow_ids,
                num_states,
                num_commodities,
                num_flows,
            )
            X_list.append(np.concatenate([x_last, x_cat], axis=1))

            y = np.column_stack(
                [
                    batch["target_value"].cpu().numpy(),
                    batch["target_weight"].cpu().numpy(),
                ]
            )
            y_list.append(y)

    return (
        np.vstack(X_list),
        np.vstack(y_list),
        fallback_feature_names(
            variant,
            num_numeric_features,
            num_states,
            num_commodities,
            num_flows,
        ),
    )
