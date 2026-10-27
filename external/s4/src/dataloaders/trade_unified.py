"""Unified S4/S4ND dataloaders that DIRECTLY call the canonical github
TradeDataPipeline (src/data/dataloader.py) instead of re-implementing data
loading. This is the single source of truth for: pinned preaggregated monthly
targets, train-only 95% sparse-combo filter (-> 918 combos),
per-combo min-max normalization fit on train, and reach-back val/test
windows. The S4 datasets here only adapt the pipeline's output into the
(x, y, z) tuple shape the S4 LightningModule expects.

The github pipeline imports only stdlib + numpy/pandas/torch (no `src.`
deps), so it loads cleanly via importlib with no package-name clash against
the S4 repo's own `src` package.

Datasets registered:
    s4_embeddings  -> per-series, ids passed to TradeIDEncoder (1-D S4)
    s4_onehot      -> per-series, one-hot appended to features (1-D S4)
"""
import importlib.util
import os
from dataclasses import fields

import torch
from torch.utils.data import DataLoader

from src.dataloaders.base import SequenceDataset

# --- load the canonical github pipeline by path (no `src` clash) ---
# Resolve the repo root from this file's location so it works on any machine /
# fresh clone (was hardcoded /root/full). dataloaders/ -> src/ -> s4/ ->
# external/ -> repo root == parents[4].
import pathlib
_REPO = pathlib.Path(__file__).resolve().parents[4]
_GH_PATH = str(_REPO / "src" / "data" / "dataloader.py")
_spec = importlib.util.spec_from_file_location("github_dataloader", _GH_PATH)
_GH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_GH)


def _resolve_data_root(data_dir):
    """Resolve one explicitly selected source root, never a fallback search.

    The old adapter tried the configured path, the repository, and the
    repository parent in turn.  On a training host that could silently select
    an unrelated ``/root/data`` snapshot even though the controller supplied a
    different DATA_DIR.  ``None`` means the checkout itself (useful for tests);
    every non-null value is authoritative and must contain the selected source.
    """
    root = _REPO if data_dir in (None, "") else pathlib.Path(data_dir)
    root = root.expanduser().resolve()
    import yaml
    with (_REPO / "config/base.yaml").open() as handle:
        data = yaml.safe_load(handle)["data"]
    if data.get("source_format") == "monthly":
        path = root / data["aggregated_csv"]
        missing = [] if path.is_file() else [str(path)]
    else:
        missing = [str(root / "data" / flow) for flow in ("imports", "exports")
                   if not (root / "data" / flow).is_dir()]
    if missing:
        raise FileNotFoundError(
            "Configured S4 data root is incomplete: " + ", ".join(missing)
        )
    return root


def _canonical_pipeline(data_dir, input_len, lag_count, aggregate=False):
    """Build the root pipeline from the root experiment's resolved contract."""
    import yaml

    data_root = _resolve_data_root(data_dir)
    config_path = _REPO / "config" / "base.yaml"
    with config_path.open() as handle:
        root_config = yaml.safe_load(handle)
    data = dict(root_config.get("data") or {})
    for key, value in (root_config.get("splits") or {}).items():
        if key in data and data[key] != value:
            raise ValueError(
                f"Conflicting root data/split value for {key}: "
                f"{data[key]!r} != {value!r}"
            )
        data[key] = value
    data.update({
        "imports_dir": str(data_root / "data" / "imports"),
        "exports_dir": str(data_root / "data" / "exports"),
        "input_len": int(input_len),
        "lag_count": int(lag_count),
        "aggregate": bool(aggregate),
    })
    if data.get("aggregated_csv"):
        data["aggregated_csv"] = str(data_root / data["aggregated_csv"])
    allowed = {field.name for field in fields(_GH.DataConfig)}
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        raise ValueError(
            f"Root base.yaml contains unsupported data settings: {unknown}"
        )
    pipe = _GH.TradeDataPipeline(_GH.DataConfig(**data))
    pipe.load_data()
    pipe.create_splits()
    return pipe


def _build_denorm(pipe):
    """(state, commodity, flow) -> (val_min, val_range, wt_min, wt_range) for
    inverse min-max, so eval can plot aggregates in real units."""
    m = {}
    for _, r in pipe.stats["exact"].iterrows():
        key = (str(r["State"]), str(r["Commodity"]), str(r["Import/Export"]))
        vr = float(r["val_max"]) - float(r["val_min"])
        wr = float(r["wt_max"]) - float(r["wt_min"])
        m[key] = (float(r["val_min"]), vr if vr else 1.0,
                  float(r["wt_min"]), wr if wr else 1.0)
    return m


class _TradePerSeriesBase(SequenceDataset):
    """Per-series S4 dataset backed by the canonical pipeline."""
    _name_ = "s4_perseries_base"
    encoding = "embeddings"  # overridden by subclasses

    @property
    def init_defaults(self):
        return {
            "input_len": 36,
            "lag_count": 12,
            "aggregate": False,    # Test 1: collapse to one summed series
        }

    def setup(self):
        if getattr(self, "_is_setup", False):
            return
        self._is_setup = True
        self.l_output = 0
        self.d_output = 2

        self.pipe = _canonical_pipeline(
            self.data_dir,
            self.input_len,
            self.lag_count,
            aggregate=getattr(self, "aggregate", False),
        )

        # One get_dataloaders call materializes id maps + the per-series
        # TradeDataset objects (with reach-back target-time windows). We
        # reuse the .dataset handles and rebuild DataLoaders per request so
        # batch_size/num_workers from the S4 loader config are honored.
        tr, va, te = self.pipe.get_dataloaders(
            batch_size=64, num_workers=0, shuffle_train=False, combo=False
        )
        self._ds = {"train": tr.dataset, "val": va.dataset, "test": te.dataset}
        self._gh_collate = tr.collate_fn

        self.n_states = len(self.pipe.state2id)
        self.n_comms = len(self.pipe.comm2id)
        self.n_flows = len(self.pipe.flow2id)
        n_feat = len(self.pipe.feat_cols)

        if self.encoding == "onehot":
            self.d_input = n_feat + self.n_states + self.n_comms + self.n_flows
        else:  # embeddings: TradeIDEncoder appends (state=8, comm=32, flow=2)
            self.d_input = n_feat + 8 + 32 + 2

        # Per-test-sample (key, time), aligned with the unshuffled test loader,
        # + denorm map -> lets s4_eval draw real-unit aggregate plots.
        id2s = {v: k for k, v in self.pipe.state2id.items()}
        id2c = {v: k for k, v in self.pipe.comm2id.items()}
        id2f = {v: k for k, v in self.pipe.flow2id.items()}
        samples = self._ds["test"].samples
        self.test_keys = [(id2s[s["state_id"]], id2c[s["comm_id"]], id2f[s["flow_id"]])
                          for s in samples]
        self.test_times = [s["target_time"] for s in samples]
        self._denorm = _build_denorm(self.pipe)

    def _collate_fn(self, batch):
        """Convert a list of github sample dicts into S4's (x, y, z) tuple."""
        gh = self._gh_collate(batch)
        x = gh["x_numeric"]                                   # (B, L, F)
        y = torch.stack([gh["target_value"], gh["target_weight"]], dim=-1)  # (B, 2)

        if self.encoding == "onehot":
            B, L, _ = x.shape
            s = torch.nn.functional.one_hot(gh["state_ids"], self.n_states).float()
            c = torch.nn.functional.one_hot(gh["comm_ids"], self.n_comms).float()
            f = torch.nn.functional.one_hot(gh["flow_ids"], self.n_flows).float()
            cat = torch.cat([s, c, f], dim=-1).unsqueeze(1).expand(-1, L, -1)
            x = torch.cat([x, cat], dim=-1)
            return x, y, {}

        return x, y, {
            "state_ids": gh["state_ids"],
            "comm_ids": gh["comm_ids"],
            "commodity_ids": gh["comm_ids"],
            "flow_ids": gh["flow_ids"],
        }

    def _dataloader(self, dataset, **kwargs):
        return DataLoader(dataset, collate_fn=self._collate_fn, **kwargs)

    def train_dataloader(self, **kwargs):
        if self.pipe.config.batch_by_target_month:
            kwargs = month_loader_kwargs(self._ds["train"].samples, kwargs)
            return self._dataloader(self._ds["train"], **kwargs)
        kwargs.setdefault("shuffle", True)
        return self._dataloader(self._ds["train"], **kwargs)

    def val_dataloader(self, **kwargs):
        kwargs["shuffle"] = False
        return self._dataloader(self._ds["val"], **kwargs)

    def test_dataloader(self, **kwargs):
        kwargs["shuffle"] = False
        return self._dataloader(self._ds["test"], **kwargs)


class TradeS4Embeddings(_TradePerSeriesBase):
    _name_ = "s4_embeddings"
    encoding = "embeddings"


class TradeS4OneHot(_TradePerSeriesBase):
    _name_ = "s4_onehot_u"
    encoding = "onehot"


def month_loader_kwargs(samples, kwargs):
    """Use the same month batches as native models, preserving S4 collation."""
    kwargs = dict(kwargs)
    if kwargs.get("drop_last", False):
        raise ValueError("Month batching cannot drop training targets")
    if kwargs.get("sampler") is not None or kwargs.get("batch_sampler") is not None:
        raise ValueError("Month batching cannot be combined with another sampler")
    sampler_type = _GH._load_sibling("src.data.sampling", "data/sampling.py").TargetMonthBatchSampler
    sampler = sampler_type(samples, shuffle=kwargs.pop("shuffle", True),
                           generator=kwargs.get("generator"))
    for key in ("batch_size", "drop_last", "sampler", "batch_sampler"):
        kwargs.pop(key, None)
    kwargs["batch_sampler"] = sampler
    return kwargs


class MonthBatchedGrid:
    """S4ND builders emit contiguous (month, subgrid) samples."""

    def train_dataloader(self, **kwargs):
        if not self.pipe.config.batch_by_target_month:
            return super().train_dataloader(**kwargs)
        times = sorted(self.pipe.df_train.Time.unique())[self.pipe.config.input_len:]
        if self.pipe.config.train_target_start is not None:
            import pandas as pd
            times = [t for t in times if t >= pd.Timestamp(self.pipe.config.train_target_start)]
        count = len(self.dataset_train)
        if not times or count % len(times):
            raise ValueError("S4ND samples do not form complete target months")
        per_month = count // len(times)
        if per_month == 0:
            raise ValueError("S4ND has no samples per target month")
        samples = [{"target_time": t} for t in times for _ in range(per_month)]
        return self._dataloader(self.dataset_train, **month_loader_kwargs(samples, kwargs))
