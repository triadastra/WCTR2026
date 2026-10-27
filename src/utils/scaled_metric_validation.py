"""Bind per-series targets, inverse transforms and metrics to pinned CSVs."""
from functools import lru_cache
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.inference.scaled_metrics import add_mase_rmsse


@lru_cache(maxsize=2)
def _source_grid(config_json, source_signature):
    # The signature invalidates the cache when a consumed file changes. The
    # loader still verifies the actual source hashes on every reconstruction.
    from src.data.dataloader import DataConfig, TradeDataPipeline
    pipe = TradeDataPipeline(DataConfig(**json.loads(config_json)))
    pipe.load_data()
    pipe.create_splits()
    return SimpleNamespace(config=pipe.config, df_metric_grid=pipe.df_metric_grid,
                           stats=pipe.stats,
                           source_fingerprint=pipe.source_fingerprint,
                           source_domain=pipe.source_domain)


def source_scaled_metrics(payload, predictions, targets, keys, *, identities,
                          predictions_real, targets_real):
    config = dict(payload["run_contract"]["data_config"])
    if not config.get("normalize", True):
        raise ValueError("seasonal metric validation requires normalized source data")
    if config.get("expected_source_fingerprint") != payload["source_fingerprint"]:
        raise ValueError("seasonal metric source fingerprint is not pinned")
    root = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2])).resolve()
    config.update(imports_dir=str(root / "data/imports"), exports_dir=str(root / "data/exports"))
    if config.get("source_format") == "monthly":
        config["aggregated_csv"] = str(root / "data/processed/hs2_monthly_aggregated.csv")
    # Seasonal scales use the full pre-burn-in training grid; model window,
    # lags and supervised target start cannot change them. Share that grid
    # across seeds and sweep coordinates without caching model-ready windows.
    config.update(input_len=1, lag_count=0, train_target_start=None)
    files = sorted(p for name in ("imports_dir", "exports_dir")
                   for p in Path(config[name]).glob("*.csv"))
    if config.get("source_format") == "monthly":
        files = [Path(config["aggregated_csv"])]
    signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    pipe = _source_grid(json.dumps(config, sort_keys=True), signature)
    if (pipe.source_fingerprint != payload["source_fingerprint"]
            or pipe.source_domain != payload["source_domain"]):
        raise ValueError("seasonal metric source differs from the run contract")

    # A matching source hash and agreement across runs cannot authenticate the
    # CSV's outcomes. Rebuild the keyed test panel from the actual source.
    grid = pipe.df_metric_grid
    test = grid[(grid.Time >= pipe.config.test_start) & (grid.Time <= pipe.config.test_end)]
    reference = {}
    for row in test.itertuples(index=False, name=None):
        values = dict(zip(test.columns, row))
        key = tuple(str(values[c]) for c in ("State", "Commodity", "Import/Export"))
        identity = (*key, values["Time"].strftime("%Y-%m-%d"))
        # All native and S4 evaluators first obtain float32 dataset targets.
        reference[identity] = np.asarray(
            [values["Value"], values["Weight"]], dtype=np.float32
        ).astype(np.float64)
    if len(identities) != len(set(identities)) or set(identities) != set(reference):
        raise ValueError("prediction target identities differ from pinned source")
    expected = np.asarray([reference[i] for i in identities])
    if not np.allclose(targets, expected, rtol=1e-9, atol=1e-8):
        raise ValueError("test targets differ from pinned source")

    stats = pipe.stats["exact"].set_index(["State", "Commodity", "Import/Export"])
    ordered = stats.loc[list(keys)]
    minimum = ordered[["val_min", "wt_min"]].to_numpy(dtype=np.float64)
    scale = ordered[["val_max", "wt_max"]].to_numpy(dtype=np.float64) - minimum
    scale[scale == 0] = 1.0
    for label, actual, normalized in (
        ("target", targets_real, expected),
        ("prediction", predictions_real, np.asarray(predictions, dtype=np.float64)),
    ):
        if not np.allclose(actual, normalized * scale + minimum, rtol=1e-9, atol=1e-8):
            raise ValueError(f"{label} inverse transforms differ from pinned source scaling")
    metrics = {}
    add_mase_rmsse(metrics, pipe, predictions, expected, keys)
    return metrics
