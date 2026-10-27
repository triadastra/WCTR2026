#!/usr/bin/env python
"""SHAP attributions for every Exp 1 (single aggregate series) run.

Read-only post-analysis of finished runs; nothing under --runs-dir is written.

Each prediction sees a 36-month window x 28 features per month (value,
weight, 12 value lags, 12 weight lags, sin/cos of the month) on the
normalized training scale. Attributions:

  * neural + baseline arms: expected-gradients SHAP (integrated gradients
    averaged over every training window as a baseline, midpoint rule), so
    phi sums to f(x) - mean f(training windows);
  * XGBoost / LightGBM: exact TreeSHAP from the libraries themselves
    (pred_contribs / pred_contrib), whose base value is the tree-path mean.

The lag columns repeat months already inside the window, so credit is split
arbitrarily between copies of the same number. Summaries therefore add phi by
*source month* (months before the target) and by *channel* (value, weight,
season) before taking magnitudes.

Every run is gated: recomputed test predictions must match its
predictions.csv, and phi must sum to the prediction minus the base value.
Failing runs are reported in Checks and left out of the per-arm tables.

Usage:
    python scripts/explain_exp1.py --runs-dir outputs/agg --out outputs/shap_exp1
"""

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataConfig, TradeDataPipeline
from src.inference.cohort import apply_checkpoint_cohort_config
from src.models import create_model
from src.utils import get_data_config, load_config, set_seed
from src.utils.config import resolve_data_paths

TREES = ("xgboost", "lightgbm", "random_forest")
BASELINES = ("moving_average", "random_walk", "seasonal_naive")
OUTPUTS = ("value", "weight")
PRED_TOL = 1e-4          # normalized units; predictions.csv is float64 text
# max |sum phi - (f - base)| over points, relative to the run's mean |f - base|
# for that output. A pointwise ratio explodes where f is within ~1e-3 of the
# base value, which says nothing about the attribution quality.
COMPLETENESS_TOL = 0.02


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True, help="Exp 1 run directories (outputs/agg)")
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--input-len", type=int, default=36)
    p.add_argument("--lag-count", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=33)
    p.add_argument("--ma-window", type=int, default=12)
    p.add_argument("--steps", type=int, default=32, help="integration steps per baseline")
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=None, help="run names or model_variant arms")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Runs and data
# --------------------------------------------------------------------------- #
def discover_runs(runs_dir, only):
    runs = []
    for metrics_path in sorted(Path(runs_dir).glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        run_dir = metrics_path.parent
        arm = f"{m['model']}_{m['variant']}"
        if only and run_dir.name not in only and arm not in only:
            continue
        runs.append({"run": run_dir.name, "dir": run_dir, "model": m["model"],
                     "variant": m["variant"], "seed": int(m["seed"]), "arm": arm})
    return runs


def checkpoint_for(run):
    if run["model"] in TREES:
        return run["dir"] / "best_xgb"
    if run["model"] in BASELINES:
        return None
    return run["dir"] / "best.pth"


def resolve_cfg(args, model, variant, checkpoint):
    """Same resolution order as scripts/inference.py for --aggregate --normalized."""
    cfg = load_config(args.config)
    for path in (f"config/models/{model}.yaml", f"config/variants/{variant}.yaml",
                 "config/exp1.yaml"):
        if Path(path).exists():
            cfg = load_config(path, cfg)
    cfg["training"]["batch_size"] = args.batch_size
    cfg["data"]["aggregate"] = True
    cfg["data"]["input_len"] = args.input_len
    cfg["data"]["lag_count"] = args.lag_count
    if checkpoint is not None:
        apply_checkpoint_cohort_config(cfg, checkpoint)
    cfg["data"] = resolve_data_paths(cfg["data"], Path(args.data_dir))
    return cfg


def build_pipeline(cfg, seed):
    set_seed(seed)
    pipeline = TradeDataPipeline(DataConfig(**get_data_config(cfg)))
    pipeline.load_data()
    pipeline.create_splits()
    loaders = pipeline.get_dataloaders(batch_size=cfg["training"]["batch_size"],
                                       num_workers=0, combo=False)
    return pipeline, loaders


def stack(dataset):
    s = dataset.samples
    return {
        "x": torch.stack([r["x_numeric"] for r in s]),
        "state": torch.tensor([r["state_id"] for r in s], dtype=torch.long),
        "comm": torch.tensor([r["comm_id"] for r in s], dtype=torch.long),
        "flow": torch.tensor([r["flow_id"] for r in s], dtype=torch.long),
        "time": [pd.Timestamp(r["target_time"]) for r in s],
    }


def construct_model(cfg, pipeline, model, variant, seed, ma_window):
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model_cfg.pop("variant", None)
    emb = model_cfg.pop("embedding_dims", None)
    if emb:
        model_cfg.setdefault("state_embed_dim", emb.get("state", 8))
        model_cfg.setdefault("comm_embed_dim", emb.get("commodity", 32))
        model_cfg.setdefault("flow_embed_dim", emb.get("flow", 2))
    if model in TREES:
        model_cfg.setdefault("random_state", seed)
    if model in BASELINES:
        model_cfg = {"window": ma_window} if model == "moving_average" else {}
    return create_model(
        name=model, variant=variant,
        num_numeric_features=len(pipeline.feat_cols),
        num_states=len(pipeline.state2id), num_commodities=len(pipeline.comm2id),
        num_flows=len(pipeline.flow2id), **model_cfg,
    )


def load_torch_weights(net, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    net.load_state_dict(state)


def saved_test_predictions(run_dir):
    df = pd.read_csv(run_dir / "predictions.csv", parse_dates=["Time"])
    return df.sort_values("Time")[["pred_value", "pred_weight"]].to_numpy(np.float64), df


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
class Forward:
    """f(x) -> (N, 2) with the run's constant aggregate IDs."""

    def __init__(self, net, ids, device, rnn):
        self.net, self.ids, self.device, self.rnn = net, ids, device, rnn

    def __call__(self, x):
        n = x.shape[0]
        s, c, f = (t[:1].expand(n).to(self.device) for t in self.ids)
        if self.rnn and self.device.type == "cuda":
            # cuDNN RNN backward refuses eval mode; the native kernel is exact.
            with torch.backends.cudnn.flags(enabled=False):
                return self.net(x, s, c, f)
        return self.net(x, s, c, f)


def expected_gradients(fwd, explain_x, background, steps, device):
    """phi (N, 2, L, F), f(x) (N, 2), base (2,) via baseline-averaged IG."""
    bg = background.to(device)
    with torch.no_grad():
        base = fwd(bg).double().mean(0)
        fx = fwd(explain_x.to(device)).double()
    alphas = ((torch.arange(steps, device=device) + 0.5) / steps).float()
    phis = []
    for x in explain_x.to(device):
        diff = (x.unsqueeze(0) - bg)                                   # (B, L, F)
        pts = (bg.unsqueeze(1) + alphas.view(1, -1, 1, 1) * diff.unsqueeze(1))
        pts = pts.reshape(-1, *x.shape).detach().requires_grad_(True)  # (B*m, L, F)
        out = fwd(pts)
        per_out = []
        for j in range(out.shape[1]):
            (g,) = torch.autograd.grad(out[:, j].sum(), pts, retain_graph=j + 1 < out.shape[1])
            g = g.view(bg.shape[0], steps, *x.shape).mean(1)           # (B, L, F)
            per_out.append((diff * g).double().mean(0))                # (L, F)
        phis.append(torch.stack(per_out))
    return torch.stack(phis).cpu().numpy(), fx.cpu().numpy(), base.cpu().numpy()


def tree_shap(wrapper, loader_list, n_numeric):
    """Exact TreeSHAP for the two per-target boosters."""
    import xgboost as xgb
    X = np.vstack([wrapper._extract_numpy(ld)[0] for ld in loader_list])
    frame = wrapper._to_frame(X)
    phis, preds, bases = [], [], []
    for reg in (wrapper.model_value, wrapper.model_weight):
        if isinstance(reg, xgb.XGBModel):
            best = getattr(reg, "best_iteration", None)
            rng = (0, best + 1) if best is not None else (0, 0)
            dm = xgb.DMatrix(frame, enable_categorical=True)
            contrib = reg.get_booster().predict(dm, pred_contribs=True, iteration_range=rng)
        else:  # LightGBM uses best_iteration_ by default
            contrib = np.asarray(reg.predict(frame, pred_contrib=True))
        preds.append(reg.predict(frame))
        phis.append(contrib[:, :-1])
        bases.append(contrib[:, -1])
    phi = np.stack(phis, axis=1).astype(np.float64)                    # (N, 2, P)
    fx = np.stack(preds, axis=1).astype(np.float64)
    base = np.stack(bases, axis=1).mean(0)
    base_rows = np.stack(bases, axis=1)
    cat = phi[:, :, n_numeric:].sum(-1)
    return phi[:, :, :n_numeric], cat, fx, base, base_rows


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def feature_layout(feat_cols, input_len):
    """Per (window step, feature): channel and months-before-target."""
    chan = np.empty((input_len, len(feat_cols)), dtype=object)
    back = np.zeros((input_len, len(feat_cols)), dtype=int)
    for step in range(input_len):
        h = input_len - step                      # last window row is t-1
        for j, col in enumerate(feat_cols):
            name = str(col).lower()
            if name.startswith("value_lag_"):
                chan[step, j], back[step, j] = "value", h + int(name.rsplit("_", 1)[1])
            elif name.startswith("weight_lag_"):
                chan[step, j], back[step, j] = "weight", h + int(name.rsplit("_", 1)[1])
            elif name in ("sin_month", "cos_month"):
                chan[step, j], back[step, j] = "season", h
            elif j == 0:
                chan[step, j], back[step, j] = "value", h
            elif j == 1:
                chan[step, j], back[step, j] = "weight", h
            else:
                raise ValueError(f"unmapped feature column {col!r}")
    return chan, back


def summarise(phi, chan, back, split_mask, cat=None):
    """Rows of grouped importance for one run: phi (N, 2, L, F)."""
    rows_c, rows_b = [], []
    for split, mask in split_mask.items():
        p = phi[mask]
        for oi, out in enumerate(OUTPUTS):
            po = p[:, oi]                                             # (n, L, F)
            groups = {c: po[:, chan == c].sum(-1) for c in ("value", "weight", "season")}
            if cat is not None:
                groups["id_features"] = cat[mask, oi]
            total = sum(np.abs(v).mean() for v in groups.values())
            for c, v in groups.items():
                rows_c.append({"split": split, "output": out, "channel": c,
                               "mean_abs_phi": float(np.abs(v).mean()),
                               "mean_phi": float(v.mean()),
                               "share": float(np.abs(v).mean() / total) if total > 0 else np.nan})
            data_chan = chan != "season"
            months = sorted(set(back[data_chan].ravel()))
            mags = {}
            for mb in months:
                sel = data_chan & (back == mb)
                mags[mb] = np.abs(po[:, sel].sum(-1)).mean()
            tot = sum(mags.values())
            for mb in months:
                rows_b.append({"split": split, "output": out, "months_back": int(mb),
                               "mean_abs_phi": float(mags[mb]),
                               "share": float(mags[mb] / tot) if tot > 0 else np.nan})
    return rows_c, rows_b


def lookback_80(df):
    df = df.sort_values("months_back")
    cum = df["share"].cumsum()
    hit = df.loc[cum >= 0.8, "months_back"]
    return int(hit.iloc[0]) if len(hit) else np.nan


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    out = Path(args.out)
    (out / "per_run").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    runs = discover_runs(args.runs_dir, args.only)
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[(r["model"], r["variant"])].append(r)
    print(f"{len(runs)} runs across {len(by_arm)} arms on {device}", flush=True)

    checks, chan_rows, back_rows = [], [], []
    for (model, variant), arm_runs in sorted(by_arm.items()):
        first = arm_runs[0]
        t0 = time.time()
        cfg = resolve_cfg(args, model, variant, checkpoint_for(first))
        pipeline, (train_ld, val_ld, test_ld) = build_pipeline(cfg, first["seed"])
        tr, va, te = stack(train_ld.dataset), stack(val_ld.dataset), stack(test_ld.dataset)
        explain_x = torch.cat([va["x"], te["x"]])
        times = va["time"] + te["time"]
        split_mask = {"test": np.array([False] * len(va["time"]) + [True] * len(te["time"])),
                      "val+test": np.ones(len(times), dtype=bool)}
        chan, back = feature_layout(pipeline.feat_cols, args.input_len)
        ids = (tr["state"], tr["comm"], tr["flow"])
        n_numeric = args.input_len * len(pipeline.feat_cols)
        print(f"[{model}/{variant}] pipeline {time.time() - t0:.0f}s; "
              f"background {len(tr['x'])}, explain {len(times)}", flush=True)

        for run in arm_runs:
            rec = {"run": run["run"], "model": model, "variant": variant, "seed": run["seed"]}
            try:
                t1 = time.time()
                net = construct_model(cfg, pipeline, model, variant, run["seed"], args.ma_window)
                cat = None
                if model in TREES:
                    net.load(str(checkpoint_for(run)))
                    phi, cat, fx, base, base_rows = tree_shap(net, [val_ld, test_ld], n_numeric)
                    phi = phi.reshape(len(times), 2, args.input_len, -1)
                    resid = phi.sum((2, 3)) + cat - (fx - base_rows)
                    method = "TreeSHAP (library, path-dependent)"
                else:
                    if model not in BASELINES:
                        load_torch_weights(net, checkpoint_for(run), device)
                    net = net.to(device).eval()
                    rnn = model in ("lstm", "gru")
                    phi, fx, base = expected_gradients(Forward(net, ids, device, rnn),
                                                       explain_x, tr["x"], args.steps, device)
                    resid = phi.sum((2, 3)) - (fx - base)
                    method = f"expected gradients ({len(tr['x'])} baselines x {args.steps} steps)"
                saved, _ = saved_test_predictions(run["dir"])
                test_fx = fx[split_mask["test"]]
                pred_diff = float(np.abs(test_fx - saved).max()) if saved.shape == test_fx.shape else np.inf
                explained = np.abs(fx - (base_rows if model in TREES else base))
                comp = float((np.abs(resid) / np.maximum(explained.mean(0), 1e-6)).max())
                pointwise = float((np.abs(resid) / np.maximum(explained, 1e-3)).max())
                ok = pred_diff <= PRED_TOL and comp <= COMPLETENESS_TOL
                rec.update(method=method, n_explained=len(times), max_pred_diff=pred_diff,
                           max_completeness_gap=comp, max_abs_residual=float(np.abs(resid).max()),
                           max_pointwise_gap=pointwise, base_value=float(np.mean(base[0])),
                           base_weight=float(np.mean(base[1])), status="ok" if ok else "FAILED",
                           seconds=round(time.time() - t1, 1))
                np.savez_compressed(out / "per_run" / f"{run['run']}.npz", phi=phi, fx=fx,
                                    base=np.asarray(base), times=np.array([str(t.date()) for t in times]),
                                    feat_cols=np.array(pipeline.feat_cols),
                                    **({"id_phi": cat} if cat is not None else {}))
                if ok:
                    c_rows, b_rows = summarise(phi, chan, back, split_mask, cat)
                    for row in c_rows + b_rows:
                        row.update(run=run["run"], model=model, variant=variant, seed=run["seed"])
                    chan_rows += c_rows
                    back_rows += b_rows
            except Exception as exc:  # one broken run must not stop the sweep
                rec.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            checks.append(rec)
            print(f"  {run['run']}: {rec['status']} pred_diff={rec.get('max_pred_diff', float('nan')):.2e} "
                  f"gap={rec.get('max_completeness_gap', float('nan')):.2e} "
                  f"{rec.get('seconds', '')}s", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_outputs(out, pd.DataFrame(checks), pd.DataFrame(chan_rows), pd.DataFrame(back_rows), args)


def write_outputs(out, checks, chan, back, args):
    keys = ["model", "variant", "split", "output"]
    arm_chan = (chan.groupby(keys + ["channel"])
                .agg(seeds=("seed", "nunique"), share_mean=("share", "mean"), share_sd=("share", "std"),
                     mean_abs_phi=("mean_abs_phi", "mean"), mean_phi=("mean_phi", "mean"))
                .reset_index()) if len(chan) else chan
    arm_back = (back.groupby(keys + ["months_back"])
                .agg(seeds=("seed", "nunique"), share_mean=("share", "mean"), share_sd=("share", "std"))
                .reset_index()) if len(back) else back
    look = []
    if len(back):
        for (m, v, s, o), g in back.groupby(keys):
            per_seed = [lookback_80(gs) for _, gs in g.groupby("seed")]
            arm = arm_back[(arm_back.model == m) & (arm_back.variant == v)
                           & (arm_back.split == s) & (arm_back.output == o)]
            top = arm.nlargest(3, "share_mean")
            look.append({"model": m, "variant": v, "split": s, "output": o,
                         "seeds": g["seed"].nunique(),
                         "months_for_80pct_median": float(np.median(per_seed)),
                         "months_for_80pct_mean_profile": lookback_80(arm.rename(columns={"share_mean": "share"})),
                         "share_last_month": float(arm.loc[arm.months_back == 1, "share_mean"].sum()),
                         "share_months_1_12": float(arm.loc[arm.months_back <= 12, "share_mean"].sum()),
                         "share_month_12": float(arm.loc[arm.months_back == 12, "share_mean"].sum()),
                         "top3_months_back": ", ".join(str(int(x)) for x in top["months_back"])})
    look = pd.DataFrame(look)
    cross = pd.DataFrame()
    if len(arm_chan):
        wide = arm_chan.pivot_table(index=["model", "variant", "split", "output"], columns="channel",
                                    values="share_mean").reset_index()
        cross = wide
    notes = pd.DataFrame({"note": [
        "SHAP for every Exp 1 run: one aggregate series, 36-month window x 28 features, normalized training scale.",
        "Explained points: 12 validation months (2024) + 5 test months (Jan-May 2025); split column separates them.",
        f"Neural and baseline arms: expected-gradients SHAP, every training window as a baseline, {args.steps} midpoint steps.",
        "XGBoost/LightGBM: exact TreeSHAP from the library (path-dependent base value = tree-path mean).",
        "Credit is summed by channel (value / weight / season) and by months before the target BEFORE taking |.|,",
        "because lag columns duplicate months already in the window.",
        "share = mean|phi| of the group / sum over groups, per run; arm tables average runs (seeds).",
        "months_for_80pct: smallest look-back (months) holding 80% of the data-channel credit.",
        "id_features (trees only): the constant state/commodity/flow columns; should be ~0.",
        "Gates per run: recomputed test predictions within 1e-4 of predictions.csv, and",
        "max |sum phi - (f - base)| <= 2% of the run's mean |f - base| (per output). Failing runs are",
        "excluded from the arm tables. max_pointwise_gap (per point, floor 1e-3) is reported only as a diagnostic.",
        "Attributions describe how each fitted model uses its inputs on 17 points of one series;",
        "they are not causal and do not rank accuracy.",
    ]})
    path = out / "shap_exp1.xlsx"
    with pd.ExcelWriter(path) as xw:
        notes.to_excel(xw, sheet_name="Notes", index=False)
        checks.to_excel(xw, sheet_name="Checks", index=False)
        cross.to_excel(xw, sheet_name="Channel_share_by_arm", index=False)
        look.to_excel(xw, sheet_name="Lookback_by_arm", index=False)
        arm_chan.to_excel(xw, sheet_name="Channel_by_arm_long", index=False)
        arm_back.to_excel(xw, sheet_name="MonthsBack_by_arm", index=False)
        chan.to_excel(xw, sheet_name="Channel_by_run", index=False)
        back.to_excel(xw, sheet_name="MonthsBack_by_run", index=False)
    print(f"Wrote {path}")
    try:
        plot(out, arm_back, arm_chan)
    except Exception:
        traceback.print_exc()
    n_ok = int((checks["status"] == "ok").sum()) if len(checks) else 0
    print(f"{n_ok}/{len(checks)} runs passed both gates")


def plot(out, arm_back, arm_chan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not len(arm_back):
        return
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    d = arm_back[arm_back.split == "val+test"]
    arms = sorted({(m, v) for m, v in zip(d.model, d.variant)})
    for oi, o in enumerate(OUTPUTS):
        mat = np.full((len(arms), int(d.months_back.max())), np.nan)
        for i, (m, v) in enumerate(arms):
            g = d[(d.model == m) & (d.variant == v) & (d.output == o)]
            mat[i, g.months_back.to_numpy() - 1] = g.share_mean.to_numpy()
        fig, ax = plt.subplots(figsize=(13, 0.38 * len(arms) + 1.6))
        # Cap the scale: the one-month baselines (share 1.0) would otherwise
        # flatten every learned model to the bottom of the colour map.
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=0.25)
        ax.set_yticks(range(len(arms)), [f"{m} / {v}" for m, v in arms], fontsize=8)
        ax.set_xticks(range(0, mat.shape[1], 3), [str(k + 1) for k in range(0, mat.shape[1], 3)])
        ax.set_xlabel("months before the target")
        ax.set_title(f"Exp 1 SHAP: share of |credit| by source month, {o} forecast (val+test, seed mean)")
        fig.colorbar(im, ax=ax, label="share (capped at 0.25)", extend="max")
        fig.tight_layout()
        fig.savefig(figdir / f"months_back_heatmap_{o}.png", dpi=150)
        plt.close(fig)
    c = arm_chan[arm_chan.split == "val+test"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.3 * len(arms) + 2), sharey=True)
    colors = {"value": "#3b75af", "weight": "#ef8636", "season": "#519e3e", "id_features": "#888888"}
    for ax, o in zip(axes, OUTPUTS):
        left = np.zeros(len(arms))
        for ch, col in colors.items():
            vals = np.array([c[(c.model == m) & (c.variant == v) & (c.output == o) & (c.channel == ch)]
                             .share_mean.sum() for m, v in arms])
            ax.barh(range(len(arms)), vals, left=left, color=col, label=ch)
            left += vals
        ax.set_title(f"{o} forecast")
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of |credit|")
    axes[0].set_yticks(range(len(arms)), [f"{m} / {v}" for m, v in arms], fontsize=8)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Exp 1 SHAP: credit by channel (val+test, seed mean)")
    fig.tight_layout()
    fig.savefig(figdir / "channel_share.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
