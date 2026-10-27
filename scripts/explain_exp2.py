#!/usr/bin/env python
"""SHAP attributions for Exp 2 (918 per-lane series) runs.

Read-only post-analysis of finished runs; nothing under --runs-dir is written.
Companion to scripts/explain_exp1.py and reuses its feature layout.

A fixed sample of target lanes (--lanes-per-tier from each training-tonnage
tier: top 10%, middle 40%, bottom 50%) is explained at every test month, for
both outputs. Every arm is explained on the same lanes and months.

  * per-lane neural arms and baselines: expected-gradients SHAP. The
    baselines are that lane's own training windows, so phi sums to
    f(x) - mean f(lane's training windows). The lane IDs are integer inputs to
    embeddings / one-hot encoders and are held fixed, not attributed.
  * all-lanes ("combo") arms: the input is the whole (month, lane, feature)
    panel, so one lane's forecast can draw on every other lane. Baselines are
    training-month panels; attributions are kept per source lane, so credit
    splits into the target lane's own history and every other lane's.
  * XGBoost / LightGBM: exact TreeSHAP from the libraries; random forest:
    shap.TreeExplainer (needs the optional ``shap`` package). Their base
    value is the tree-path mean over ALL lanes, so lane-identity columns get
    the credit for a lane's level; that credit is reported separately.

Integration uses Gauss-Legendre nodes on [0, 1] (--steps-*), which is far
more accurate than the midpoint rule at the same number of model calls.

Every run is gated: recomputed predictions for the explained lanes must match
its predictions.csv, and credit must sum to f(x) - base within 2% of the
run's mean |f(x) - base|. Failing runs are excluded from the arm tables.

Two phases so a stopped container loses nothing: ``run`` writes one .npz per
run (and skips runs already written); ``summarise`` builds the workbook and
figures from whatever .npz files exist.

Usage:
    python scripts/explain_exp2.py run --runs-dir outputs/full --out outputs/shap_exp2
    python scripts/explain_exp2.py summarise --out outputs/shap_exp2
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from explain_exp1 import (  # noqa: E402
    BASELINES, OUTPUTS, PRED_TOL, TREES, feature_layout, load_torch_weights, lookback_80,
)
from post_inference_report import LANE, TIERS, assign_tiers, lane_sizes  # noqa: E402

COMPLETENESS_TOL = 0.02
# Other lanes grouped by the lattice axis that links them to the target lane:
# differing in exactly one coordinate makes a lane a neighbour along that axis
# (state / commodity / flow); differing in two or three is off-axis.
RELATIONS = ("state_axis", "commodity_axis", "flow_axis", "off_axis")
_AXIAL = ("cross_attention_2d", "cross_attention_3d", "cross_attention_4d",
          "matched_axial_2d", "matched_axial_3d", "matched_axial_4d")
# Heavier kernels need smaller panel chunks to stay inside GPU memory.
COMBO_CHUNK = {"mamba": 8, "mamba_nd": 16}


def is_combo(model, variant):
    """Same predicate as scripts/inference.py."""
    return ((model in ("lstm", "gru") and variant in
             ("cross_attention", "film_attention", "identity_cross_attention") + _AXIAL)
            or (model in ("transformer", "gpt", "transformer_linhead", "gpt_linhead",
                          "mamba", "s4", "s4_bidir") and variant in _AXIAL)
            or (model == "mamba_nd" and variant in _AXIAL))


def run_kind(run_name, model, variant):
    if run_name.startswith("s4_s4"):
        return "hydra_s4"          # external/s4 arms: see explain_exp2_s4.py
    if model in TREES:
        return "tree"
    if model in BASELINES:
        return "baseline"
    return "combo" if is_combo(model, variant) else "flat"


def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


# --------------------------------------------------------------------------- #
# Configuration, pipeline, lanes
# --------------------------------------------------------------------------- #
def resolve_cfg(model, variant, checkpoint, data_dir):
    """Same resolution as scripts/inference.py for Exp 2 (--normalized)."""
    from src.inference.cohort import apply_checkpoint_cohort_config
    from src.utils import load_config
    from src.utils.config import resolve_data_paths
    cfg = load_config("config/base.yaml")
    for path in (f"config/models/{model}.yaml", f"config/variants/{variant}.yaml"):
        if Path(path).exists():
            cfg = load_config(path, cfg)
    if checkpoint is not None:
        apply_checkpoint_cohort_config(cfg, checkpoint)
    cfg["data"] = resolve_data_paths(cfg["data"], Path(data_dir))
    return cfg


class Pipelines:
    """Cache one built pipeline per data configuration."""

    def __init__(self):
        self.cache = {}

    def get(self, cfg, seed, combo):
        from src.data import DataConfig, TradeDataPipeline
        from src.utils import get_data_config, set_seed
        dc = get_data_config(cfg)
        key = (json.dumps(dc, sort_keys=True, default=str), combo)
        if key not in self.cache:
            set_seed(seed)
            pipe = TradeDataPipeline(DataConfig(**dc))
            pipe.load_data()
            pipe.create_splits()
            loaders = pipe.get_dataloaders(batch_size=cfg["training"]["batch_size"],
                                           num_workers=0, shuffle_train=False, combo=combo)
            self.cache[key] = (pipe, loaders)
        return self.cache[key]


def select_lanes(pipe, data_csv, per_tier, train_start, train_end, lanes_file=None):
    """Target lanes: every kept lane (per_tier None), a user-supplied list, or
    an even spread of per_tier lanes from each training-tonnage tier."""
    size = assign_tiers(lane_sizes(data_csv, train_start, train_end))
    kept = pd.DataFrame(pipe.kept_combos, columns=LANE)
    size = kept.merge(size, on=LANE, how="left", validate="one_to_one")
    if size["train_tonnage_kg"].isna().any():
        raise ValueError("kept lanes missing from the data CSV")
    size = assign_tiers(size)          # tiers over the kept cohort only
    size = size.sort_values("train_tonnage_kg", ascending=False).reset_index(drop=True)
    if lanes_file is not None:
        wanted = pd.read_csv(lanes_file)[LANE].astype(str)
        chosen = size.merge(wanted, on=LANE, how="inner")
        if len(chosen) != len(wanted):
            raise ValueError(f"{len(wanted) - len(chosen)} requested lanes are not in the kept cohort")
        return chosen
    if per_tier is None:
        return size
    picks = []
    for tier in TIERS:
        t = size[size.size_tier == tier].reset_index(drop=True)
        idx = np.unique(np.linspace(0, len(t) - 1, per_tier).round().astype(int))
        picks.append(t.iloc[idx])
    return pd.concat(picks, ignore_index=True)


def lane_key(row):
    return (str(row["State"]), str(row["Commodity"]), str(row["Import/Export"]))


def relation(target, other):
    differs = [a != b for a, b in zip(target, other)]   # (state, commodity, flow)
    if sum(differs) == 1:
        return RELATIONS[differs.index(True)]
    return "off_axis"


def saved_predictions(run_dir, lanes, times):
    """(len(lanes), len(times), 2) normalized predictions from the run."""
    p = run_dir / "predictions_norm.csv"
    df = pd.read_csv(p if p.exists() else run_dir / "predictions.csv", parse_dates=["Time"])
    idx = {(str(s), str(c), str(f), pd.Timestamp(t)): (v, w) for s, c, f, t, v, w in zip(
        df["State"], df["Commodity"], df["Import/Export"], df["Time"], df["pred_value"], df["pred_weight"])}
    out = np.full((len(lanes), len(times), 2), np.nan)
    for i, k in enumerate(lanes):
        for j, t in enumerate(times):
            if (*k, t) in idx:
                out[i, j] = idx[(*k, t)]
    return out


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(cfg, pipe, model, variant, seed, combo):
    from src.models import create_model
    mc = dict(cfg["model"])
    mc.pop("name", None)
    mc.pop("variant", None)
    emb = mc.pop("embedding_dims", None)
    if emb:
        mc.setdefault("state_embed_dim", emb.get("state", 8))
        mc.setdefault("comm_embed_dim", emb.get("commodity", 32))
        mc.setdefault("flow_embed_dim", emb.get("flow", 2))
    if model in TREES:
        mc.setdefault("random_state", seed)
    if model in BASELINES:
        mc = {"window": 12} if model == "moving_average" else {}
    if combo:
        mc.update(num_combos=pipe.num_combos, features_per_group=len(pipe.feat_cols),
                  combo_coords=pipe.combo_coords, lattice_dims=pipe.lattice_dims)
    return create_model(name=model, variant=variant, num_numeric_features=len(pipe.feat_cols),
                        num_states=len(pipe.state2id), num_commodities=len(pipe.comm2id),
                        num_flows=len(pipe.flow2id), **mc)


def rnn_guard(model, device):
    """cuDNN RNN backward refuses eval mode; the native kernel is exact."""
    if model in ("lstm", "gru") and device.type == "cuda":
        return torch.backends.cudnn.flags(enabled=False)
    return torch.enable_grad()


# --------------------------------------------------------------------------- #
# Attribution per kind. Each returns dict(own=(N_l, N_t, 2, L, F),
# fx=(N_l, N_t, 2), base=(N_l, N_t, 2)) plus kind-specific extras.
# --------------------------------------------------------------------------- #
def flat_samples_by_lane(dataset, pipe):
    id2s = {v: k for k, v in pipe.state2id.items()}
    id2c = {v: k for k, v in pipe.comm2id.items()}
    id2f = {v: k for k, v in pipe.flow2id.items()}
    out = {}
    for s in dataset.samples:
        key = (str(id2s[s["state_id"]]), str(id2c[s["comm_id"]]), str(id2f[s["flow_id"]]))
        out.setdefault(key, []).append(s)
    for v in out.values():
        v.sort(key=lambda r: r["target_time"])
    return out


def explain_flat(net, model, pipe, loaders, lanes, times, device, bg_n, steps):
    train = flat_samples_by_lane(loaders[0].dataset, pipe)
    test = flat_samples_by_lane(loaders[2].dataset, pipe)
    nodes, weights = gauss_legendre(steps)
    a = torch.tensor(nodes, dtype=torch.float32, device=device)
    w = torch.tensor(weights, dtype=torch.float64)          # float64 sums stay on the CPU (MPS has no float64)
    L, F = loaders[2].dataset.samples[0]["x_numeric"].shape
    own = np.zeros((len(lanes), len(times), 2, L, F))
    fx = np.zeros((len(lanes), len(times), 2))
    base = np.zeros_like(fx)
    for li, key in enumerate(lanes):
        by_time = {pd.Timestamp(s["target_time"]): s for s in test[key]}
        xs = torch.stack([by_time[t]["x_numeric"] for t in times]).to(device)          # (T, L, F)
        tr = train[key]
        pick = np.unique(np.linspace(0, len(tr) - 1, min(bg_n, len(tr))).round().astype(int))
        bg = torch.stack([tr[i]["x_numeric"] for i in pick]).to(device)                # (B, L, F)
        s0 = tr[0]
        ids = [torch.tensor([s0[k]], device=device) for k in ("state_id", "comm_id", "flow_id")]

        def f(x):
            n = x.shape[0]
            return net(x, *(t.expand(n) for t in ids))

        with rnn_guard(model, device):
            with torch.no_grad():
                fx[li] = f(xs).cpu().double().numpy()
                base[li] = f(bg).cpu().double().mean(0).numpy()
            diff = xs[:, None] - bg[None]                                              # (T, B, L, F)
            pts = bg[None, :, None] + a.view(1, 1, -1, 1, 1) * diff[:, :, None]         # (T, B, m, L, F)
            pts = pts.reshape(-1, L, F).detach().requires_grad_(True)
            out = f(pts)
            for j in range(2):
                (g,) = torch.autograd.grad(out[:, j].sum(), pts, retain_graph=j == 0)
                g = g.view(len(times), len(pick), steps, L, F).cpu().double()
                ig = (g * w.view(1, 1, -1, 1, 1)).sum(2) * diff.cpu().double()         # (T, B, L, F)
                own[li, :, j] = ig.mean(1).numpy()
    return {"own": own, "fx": fx, "base": base}


def explain_tree(wrapper, model, pipe, loaders, lanes, times, n_numeric, rf_workers=1):
    """Exact TreeSHAP; returns own (numeric window) and id (lane-identity columns)."""
    ds = loaders[2].dataset
    X, _ = wrapper._extract_numpy(loaders[2])
    id2s = {v: k for k, v in pipe.state2id.items()}
    id2c = {v: k for k, v in pipe.comm2id.items()}
    id2f = {v: k for k, v in pipe.flow2id.items()}
    row = {}
    for i, s in enumerate(ds.samples):
        key = (str(id2s[s["state_id"]]), str(id2c[s["comm_id"]]), str(id2f[s["flow_id"]]))
        row[(key, pd.Timestamp(s["target_time"]))] = i
    rows = [row[(k, t)] for k in lanes for t in times]
    # Random forest was fit on the raw matrix; the boosters on a categorical frame.
    frame = X[rows] if model == "random_forest" else wrapper._to_frame(X[rows])
    phis, preds, bases = [], [], []
    for reg in (wrapper.model_value, wrapper.model_weight):
        if model == "xgboost":
            import xgboost as xgb
            best = getattr(reg, "best_iteration", None)
            rng = (0, best + 1) if best is not None else (0, 0)
            # GPU TreeSHAP is inexact on categorical splits (off by ~5e-3 on the
            # embeddings arm); the CPU path sums to the prediction to ~1e-7.
            booster = reg.get_booster()
            booster.set_param({"device": "cpu"})
            contrib = booster.predict(xgb.DMatrix(frame, enable_categorical=True),
                                      pred_contribs=True, iteration_range=rng)
            phi, b = contrib[:, :-1], contrib[:, -1]
        elif model == "lightgbm":
            contrib = np.asarray(reg.predict(frame, pred_contrib=True))
            phi, b = contrib[:, :-1], contrib[:, -1]
        else:
            phi, b = forest_shap(reg, np.asarray(frame, np.float64), rf_workers)
        phis.append(phi)
        preds.append(reg.predict(frame))
        bases.append(b)
    phi = np.stack(phis, 1).astype(np.float64)                                       # (R, 2, P)
    nl, nt = len(lanes), len(times)
    L = ds.input_len
    own = phi[:, :, :n_numeric].reshape(nl, nt, 2, L, -1)
    ids = phi[:, :, n_numeric:].sum(-1).reshape(nl, nt, 2)
    fx = np.stack(preds, 1).astype(np.float64).reshape(nl, nt, 2)
    base = np.stack(bases, 1).astype(np.float64).reshape(nl, nt, 2)
    return {"own": own, "id": ids, "fx": fx, "base": base}


_FOREST = None


def _forest_chunk(X):
    return np.asarray(_FOREST.shap_values(X, check_additivity=False))


def forest_shap(reg, X, workers):
    """shap.TreeExplainer on a scikit-learn forest. Its tree walk is single
    threaded, so rows are split over forked workers that share the loaded
    forest copy-on-write (the production forests are ~1.3 GB each)."""
    global _FOREST
    import multiprocessing as mp
    import shap
    _FOREST = shap.TreeExplainer(reg, feature_perturbation="tree_path_dependent")
    base = np.full(len(X), float(np.ravel(_FOREST.expected_value)[0]))
    workers = max(1, min(workers, len(X)))
    if workers == 1:
        return _forest_chunk(X), base
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_forest_chunk, np.array_split(X, workers))
    return np.concatenate(parts), base


def explain_combo(net, model, pipe, loaders, lanes, times, device, bg_n, steps):
    """Per-source-lane attributions for all-lanes models."""
    tr_ds, te_ds = loaders[0].dataset, loaders[2].dataset
    g_of = {tuple(map(str, k)): g for k, g in pipe.combo2id.items()}
    targets = [g_of[k] for k in lanes]
    t_items = {pd.Timestamp(te_ds[i]["target_time"]): i for i in range(len(te_ds))}
    pick = np.unique(np.linspace(0, len(tr_ds) - 1, min(bg_n, len(tr_ds))).round().astype(int))
    bg = torch.stack([tr_ds[int(i)]["x_numeric"] for i in pick]).to(device)          # (B, L, G, F)
    B, L, G, F = bg.shape
    nodes, weights = gauss_legendre(steps)
    chunk = COMBO_CHUNK.get(model, 32)
    own = np.zeros((len(lanes), len(times), 2, L, F))
    src = np.zeros((len(lanes), len(times), 2, G, 3))                                  # value/weight/season
    fx = np.zeros((len(lanes), len(times), 2))
    base = np.zeros_like(fx)
    chan_idx = channel_index(pipe.feat_cols)
    tg = torch.tensor(targets, device=device)
    # MPS has no float64: there the float64 sums run on the CPU. CUDA/CPU keep
    # accumulating on `device` exactly as before.
    acc_dev = torch.device("cpu") if device.type == "mps" else device
    for ti, t in enumerate(times):
        item = te_ds[t_items[t]]
        x = item["x_numeric"].to(device)
        gm = item["group_mask"].to(device)

        def f(p):
            return net(p, None, None, None, group_mask=gm.expand(p.shape[0], -1))

        with rnn_guard(model, device):
            with torch.no_grad():
                fx[:, ti] = f(x[None])[0, tg].to(acc_dev).double().cpu().numpy()
                bo = torch.cat([f(bg[i:i + chunk]) for i in range(0, B, chunk)])
                base[:, ti] = bo[:, tg].to(acc_dev).double().mean(0).cpu().numpy()
            acc = torch.zeros((len(lanes), 2, L, G, F), dtype=torch.float64, device=acc_dev)
            combos = [(b, k) for b in range(B) for k in range(steps)]
            for c0 in range(0, len(combos), chunk):
                part = combos[c0:c0 + chunk]
                bi = torch.tensor([b for b, _ in part], device=device)
                ak = torch.tensor([nodes[k] for _, k in part], dtype=torch.float32, device=device)
                wk = torch.tensor([weights[k] for _, k in part], dtype=torch.float64, device=acc_dev)
                diff = x[None] - bg[bi]                                               # (K, L, G, F)
                diff64 = diff.to(acc_dev).double()
                pts = (bg[bi] + ak.view(-1, 1, 1, 1) * diff).detach().requires_grad_(True)
                out = f(pts)
                n_t = len(targets) * 2
                for q, (li, j) in enumerate((li, j) for li in range(len(targets)) for j in range(2)):
                    (g,) = torch.autograd.grad(out[:, targets[li], j].sum(), pts,
                                               retain_graph=q + 1 < n_t)
                    acc[li, j] += ((g.to(acc_dev).double() * diff64) * wk.view(-1, 1, 1, 1)).sum(0) / B
                del out, pts
            a = acc.cpu().numpy()                                                    # (Nl, 2, L, G, F)
        for li, g in enumerate(targets):
            own[li, ti] = a[li, :, :, g, :]
            for c in range(3):
                src[li, ti, :, :, c] = a[li][..., chan_idx == c].sum(axis=(1, 3))   # (2, G)
            src[li, ti, :, g, :] = 0.0                     # own lane lives in `own`
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combo_keys = [None] * G
    for k, g in pipe.combo2id.items():
        combo_keys[g] = "|".join(map(str, k))
    return {"own": own, "src": src, "fx": fx, "base": base, "targets": np.array(targets),
            "combo_keys": np.array(combo_keys)}


def channel_index(feat_cols):
    """0 value, 1 weight, 2 season for each per-month feature column."""
    chan, _ = feature_layout(feat_cols, 1)
    return np.array([{"value": 0, "weight": 1, "season": 2}[c] for c in chan[0]])


# --------------------------------------------------------------------------- #
# Run phase
# --------------------------------------------------------------------------- #
def discover(runs_dir, kinds, combo_seeds, only, skip_models=(), seeds=()):
    runs = []
    for mp in sorted(Path(runs_dir).glob("*/metrics.json")):
        m = json.loads(mp.read_text())
        name = mp.parent.name
        kind = run_kind(name, m["model"], m["variant"])
        if kind not in kinds:
            continue
        seed = int(m["seed"])
        if seeds and seed not in seeds:
            continue
        if kind == "combo" and combo_seeds and seed not in combo_seeds:
            continue
        arm = f"{m['model']}_{m['variant']}"
        if m["model"] in skip_models:
            continue
        if only and name not in only and arm not in only:
            continue
        runs.append(dict(run=name, dir=mp.parent, model=m["model"], variant=m["variant"],
                         seed=seed, kind=kind))
    cost = {"baseline": 0, "flat": 1, "tree": 2, "combo": 3}
    heavy = {"mamba": 1, "mamba_nd": 2}
    runs.sort(key=lambda r: (cost[r["kind"]], heavy.get(r["model"], 0), r["run"]))
    return runs


def cmd_run(args):
    out = Path(args.out)
    (out / "per_run").mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    kinds = set(args.kinds)
    runs = discover(args.runs_dir, kinds, set(args.combo_seeds), args.only, set(args.skip_models),
                    set(args.seeds or []))
    print(f"{len(runs)} runs ({', '.join(sorted(kinds))}) on {device}", flush=True)
    pipes = Pipelines()
    lanes_df = None
    log = out / "checks.jsonl"
    for r in runs:
        dest = out / "per_run" / f"{r['run']}.npz"
        if dest.exists() and not args.overwrite and str(np.load(dest)["status"]) == "ok":
            print(f"  {r['run']}: done, skipped", flush=True)
            continue
        rec = {k: r[k] for k in ("run", "model", "variant", "seed", "kind")}
        t0 = time.time()
        try:
            ckpt = None if r["kind"] == "baseline" else (
                r["dir"] / ("best_xgb" if r["kind"] == "tree" else "best.pth"))
            combo = r["kind"] == "combo"
            cfg = resolve_cfg(r["model"], r["variant"], ckpt, args.data_dir)
            pipe, loaders = pipes.get(cfg, r["seed"], combo)
            if lanes_df is None:
                lanes_df = select_lanes(pipe, args.data_csv,
                                        None if args.all_lanes else args.lanes_per_tier,
                                        args.train_start, args.train_end, args.lanes_file)
                lanes_df.to_csv(out / "lanes.csv", index=False)
            lanes = [lane_key(x) for _, x in lanes_df.iterrows()]
            times = sorted({pd.Timestamp(t) for t in pd.date_range(args.test_start, args.test_end,
                                                                     freq="MS")})
            net = build_model(cfg, pipe, r["model"], r["variant"], r["seed"], combo)
            if r["kind"] == "tree":
                net.load(str(ckpt))
                res = explain_tree(net, r["model"], pipe, loaders, lanes, times,
                                   pipe.config.input_len * len(pipe.feat_cols), args.rf_workers)
            else:
                if r["kind"] != "baseline":
                    load_torch_weights(net, ckpt, device)
                net = net.to(device).eval()
                fn, bg_n, steps = ((explain_combo, args.bg_combo, args.steps_combo) if combo
                                   else (explain_flat, args.bg_flat, args.steps_flat))
                res = fn(net, r["model"], pipe, loaders, lanes, times, device, bg_n, steps)
                check(res, r, lanes, times, rec)
                used = steps
                for factor in (2, 4):
                    if rec["status"] == "ok" or rec["max_pred_diff"] > PRED_TOL:
                        break
                    # Predictions reproduce, so only the quadrature is short:
                    # retry with 2x, then 4x the Gauss-Legendre nodes.
                    rec.setdefault("retries", []).append({"steps": used, "gap": rec["max_completeness_gap"]})
                    used = factor * steps
                    res = fn(net, r["model"], pipe, loaders, lanes, times, device, bg_n, used)
                    check(res, r, lanes, times, rec)
                rec["steps"] = used
            if r["kind"] == "tree":
                check(res, r, lanes, times, rec)
            save(res, lanes, times, pipe, dest, rec)
        except Exception as exc:  # one broken run must not stop the sweep
            rec.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        rec["seconds"] = round(time.time() - t0, 1)
        with open(log, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"  {r['run']}: {rec.get('status')} pred_diff={rec.get('max_pred_diff', np.nan):.2e} "
              f"gap={rec.get('max_completeness_gap', np.nan):.2e} {rec['seconds']}s", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def check(res, r, lanes, times, rec):
    saved = saved_predictions(r["dir"], lanes, times)
    pred_diff = float(np.nanmax(np.abs(res["fx"] - saved))) if np.isfinite(saved).all() else np.inf
    total = res["own"].sum((-2, -1))
    if "src" in res:
        total = total + res["src"].sum((-2, -1))
    if "id" in res:
        total = total + res["id"]
    explained = np.abs(res["fx"] - res["base"])
    resid = total - (res["fx"] - res["base"])
    scale = np.maximum(explained.reshape(-1, 2).mean(0), 1e-6)
    comp = float((np.abs(resid).reshape(-1, 2) / scale).max())
    ok = pred_diff <= PRED_TOL and comp <= COMPLETENESS_TOL
    rec.update(status="ok" if ok else "FAILED", max_pred_diff=pred_diff,
               max_completeness_gap=comp, max_abs_residual=float(np.abs(resid).max()),
               n_lanes=len(lanes), n_months=len(times))


def save(res, lanes, times, pipe, dest, rec):
    extra = {k: v for k, v in res.items() if k not in ("own", "fx", "base")}
    np.savez_compressed(dest, own=res["own"].astype(np.float32), fx=res["fx"], base=res["base"],
                        lanes=np.array(["|".join(k) for k in lanes]),
                        times=np.array([str(t.date()) for t in times]),
                        feat_cols=np.array(pipe.feat_cols), status=rec["status"],
                        **{k: (v.astype(np.float32) if k == "src" else v) for k, v in extra.items()})


# --------------------------------------------------------------------------- #
# Summarise phase
# --------------------------------------------------------------------------- #
def point_groups(z, chan, back):
    """Net credit per group for every (lane, month, output) point."""
    own = z["own"].astype(np.float64)                                   # (Nl, Nt, 2, L, F)
    g = {"own_value": own[..., chan == "value"].sum(-1),
         "own_weight": own[..., chan == "weight"].sum(-1),
         "season": own[..., chan == "season"].sum(-1)}
    if "src" in z.files:
        src = z["src"].astype(np.float64)                               # (Nl, Nt, 2, G, 3)
        lanes = [tuple(s.split("|")) for s in z["lanes"]]
        combo_keys = [tuple(s.split("|")) for s in z["combo_keys"]]
        rel = np.array([[relation(key, other) for other in combo_keys] for key in lanes])
        for name in RELATIONS:
            mask = np.stack([rel[li] == name for li in range(len(lanes))])     # (Nl, G)
            m = mask[:, None, None, :, None]
            g[f"other_{name}"] = (src[..., :2] * m).sum((-2, -1))
        g["season"] = g["season"] + src[..., 2].sum(-1)
        g["_other_gross"] = np.abs(src[..., :2].sum(-1)).sum(-1)
        # Mean |net credit| per SOURCE lane of each relation: the relation
        # classes hold 1 to ~800 lanes, so shares alone mostly measure size.
        lane_abs = np.abs(src[..., :2].sum(-1))                          # (Nl, Nt, 2, G)
        for name in RELATIONS:
            mask = np.stack([rel[li] == name for li in range(len(lanes))])
            n = np.maximum(mask.sum(1), 1)[:, None, None]
            g[f"_perlane_{name}"] = np.where(mask.sum(1)[:, None, None] > 0,
                                             (lane_abs * mask[:, None, None, :]).sum(-1) / n, np.nan)
    if "id" in z.files:
        g["lane_identity"] = z["id"]
    return g


def summarise_run(z, chan, back, tiers):
    rows_g, rows_b = [], []
    g = point_groups(z, chan, back)
    own = z["own"].astype(np.float64)
    data = chan != "season"
    months = sorted(set(back[data].ravel()))
    for split, lane_mask in [("all", np.ones(len(tiers), bool))] + [
            (t, tiers == t) for t in TIERS]:
        for oi, out in enumerate(OUTPUTS):
            mags = {k: np.abs(v[lane_mask, :, oi]).mean() for k, v in g.items()
                    if not k.startswith("_")}
            tot = sum(mags.values())
            for k, v in mags.items():
                rows_g.append({"lanes": split, "output": out, "group": k,
                               "mean_abs_phi": float(v), "share": float(v / tot) if tot else np.nan})
            if "_other_gross" in g:
                own_net = np.abs(g["own_value"] + g["own_weight"])[lane_mask, :, oi].mean()
                gross = g["_other_gross"][lane_mask, :, oi].mean()
                rows_g.append({"lanes": split, "output": out, "group": "other_lanes_gross",
                               "mean_abs_phi": float(gross),
                               "share": float(gross / (gross + own_net)) if gross + own_net else np.nan})
            if "_perlane_off_axis" in g:
                ref = np.nanmean(g["_perlane_off_axis"][lane_mask, :, oi])
                for name in RELATIONS[:-1]:
                    v = np.nanmean(g[f"_perlane_{name}"][lane_mask, :, oi])
                    rows_g.append({"lanes": split, "output": out, "group": f"affinity_{name}",
                                   "mean_abs_phi": float(v), "share": float(v / ref) if ref else np.nan})
            po = own[lane_mask, :, oi]
            prof = {mb: np.abs(po[..., data & (back == mb)].sum(-1)).mean() for mb in months}
            t = sum(prof.values())
            for mb in months:
                rows_b.append({"lanes": split, "output": out, "months_back": int(mb),
                               "share": float(prof[mb] / t) if t else np.nan})
    return rows_g, rows_b


def cmd_summarise(args):
    out = Path(args.out)
    checks = [json.loads(line) for line in open(out / "checks.jsonl")] if (out / "checks.jsonl").exists() else []
    checks = pd.DataFrame(checks).drop_duplicates("run", keep="last") if checks else pd.DataFrame()
    lanes_df = pd.read_csv(out / "lanes.csv")
    tier_of = {"|".join(map(str, lane_key(r))): r["size_tier"] for _, r in lanes_df.iterrows()}
    G, B = [], []
    for npz in sorted((out / "per_run").glob("*.npz")):
        z = np.load(npz, allow_pickle=True)
        if str(z["status"]) != "ok":
            continue
        run = npz.stem
        meta = checks.set_index("run").loc[run] if len(checks) and run in set(checks.run) else None
        chan, back = feature_layout(list(z["feat_cols"]), z["own"].shape[3])
        tiers = np.array([tier_of[s] for s in z["lanes"]])
        rg, rb = summarise_run(z, chan, back, tiers)
        info = {"run": run, "model": meta["model"], "variant": meta["variant"],
                "seed": int(meta["seed"]), "kind": meta["kind"]}
        G += [{**info, **x} for x in rg]
        B += [{**info, **x} for x in rb]
    G, B = pd.DataFrame(G), pd.DataFrame(B)
    keys = ["kind", "model", "variant", "lanes", "output"]
    arm_g = (G.groupby(keys + ["group"]).agg(seeds=("seed", "nunique"), share_mean=("share", "mean"),
                                             share_sd=("share", "std")).reset_index())
    arm_b = (B.groupby(keys + ["months_back"]).agg(seeds=("seed", "nunique"),
                                                   share_mean=("share", "mean")).reset_index())
    wide = arm_g.pivot_table(index=keys, columns="group", values="share_mean").reset_index()
    look = []
    for k, g in arm_b.groupby(keys):
        look.append(dict(zip(keys, k), months_for_80pct=lookback_80(g.rename(columns={"share_mean": "share"})),
                         share_months_1_12=float(g.loc[g.months_back <= 12, "share_mean"].sum()),
                         share_last_month=float(g.loc[g.months_back == 1, "share_mean"].sum())))
    look = pd.DataFrame(look)
    notes = pd.DataFrame({"note": [
        "SHAP for Exp 2 runs on a fixed sample of target lanes (lanes.csv) at every test month (Jan-May 2025).",
        "Groups (net credit, summed before |.|): own_value / own_weight = the target lane's own past value / tonnage",
        "(incl. its lag columns); season = month-of-year inputs; other_* = other lanes' value+tonnage history by relation",
        "to the target along the lattice axes: state_axis / commodity_axis / flow_axis = differs in that coordinate only, off_axis = differs in 2-3;",
        "lane_identity (trees only) = state/commodity/flow columns.",
        "other_lanes_gross = sum over other lanes of |net lane credit| vs the own-lane net credit (combo arms only).",
        "affinity_<axis> = mean |credit| per neighbour on that axis / the same for an off-axis lane",
        "(in the share columns; >1 means the model leans on that axis's neighbours more than on off-axis lanes).",
        "share = group mean|phi| / sum over groups within a run; arm tables average runs (seeds).",
        "Per-lane neural arms and baselines use that lane's training windows as baselines; combo arms use training",
        "panels. Trees use the tree-path mean over all lanes, so their lane_identity absorbs lane level.",
        "Integration: Gauss-Legendre. Gate: predictions within 1e-4 and completeness within 2% of mean |f - base|.",
        "Attributions describe how each fitted model uses its inputs; they are not causal and do not rank accuracy.",
    ]})
    path = out / "shap_exp2.xlsx"
    with pd.ExcelWriter(path) as xw:
        notes.to_excel(xw, sheet_name="Notes", index=False)
        checks.to_excel(xw, sheet_name="Checks", index=False)
        lanes_df.to_excel(xw, sheet_name="Lanes", index=False)
        wide.to_excel(xw, sheet_name="Group_share_by_arm", index=False)
        look.to_excel(xw, sheet_name="Lookback_by_arm", index=False)
        arm_g.to_excel(xw, sheet_name="Groups_by_arm_long", index=False)
        arm_b.to_excel(xw, sheet_name="MonthsBack_by_arm", index=False)
        G.to_excel(xw, sheet_name="Groups_by_run", index=False)
    print(f"Wrote {path}: {G.run.nunique() if len(G) else 0} runs")
    try:
        plot(out, wide)
    except Exception:
        traceback.print_exc()


def plot(out, wide):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = wide[wide.lanes == "all"].copy()
    d["arm"] = d.model + " / " + d.variant
    order = ["own_value", "own_weight", "season"] + [f"other_{r}" for r in RELATIONS] + ["lane_identity"]
    colors = ["#3b75af", "#ef8636", "#519e3e", "#8c564b", "#c49c94", "#9467bd", "#c5b0d5", "#888888"]
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    arms = sorted(set(d.arm), key=lambda a: (d.loc[d.arm == a, "kind"].iloc[0], a))
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.28 * len(arms) + 2), sharey=True)
    for ax, o in zip(axes, OUTPUTS):
        left = np.zeros(len(arms))
        for grp, col in zip(order, colors):
            if grp not in d:
                continue
            vals = np.array([np.nan_to_num(d.loc[(d.arm == a) & (d.output == o), grp].sum()) for a in arms])
            ax.barh(range(len(arms)), vals, left=left, color=col, label=grp)
            left += vals
        ax.set_xlim(0, 1)
        ax.set_title(f"{o} forecast")
        ax.set_xlabel("share of |credit|")
    axes[0].set_yticks(range(len(arms)), arms, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Exp 2 SHAP: where each forecast's credit comes from (sampled lanes, test months)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(figdir / "group_share.png", dpi=150)
    plt.close(fig)

    aff = [f"affinity_{r}" for r in RELATIONS[:-1]]
    c = d[d[aff[0]].notna()] if aff[0] in d else d.iloc[:0]
    if len(c):
        arms = sorted(set(c.arm))
        fig, axes = plt.subplots(1, 2, figsize=(12, 0.3 * len(arms) + 2), sharey=True)
        names = {"affinity_state_axis": "state-axis neighbours (state differs)",
                 "affinity_commodity_axis": "commodity-axis neighbours (commodity differs)",
                 "affinity_flow_axis": "flow-axis neighbour (flow differs)"}
        for ax, o in zip(axes, OUTPUTS):
            y = np.arange(len(arms))
            for k, (col, colr) in enumerate(zip(aff, ["#8c564b", "#c49c94", "#9467bd"])):
                vals = [c.loc[(c.arm == a) & (c.output == o), col].sum() for a in arms]
                # Bars grow from 1 (no preference); 1e-3 marks lanes the model cannot see.
                v = np.maximum(vals, 1e-3)
                ax.barh(y + (k - 1) * 0.27, v - 1, left=1, height=0.27, color=colr, label=names[col])
            ax.axvline(1, color="k", lw=0.8)
            ax.set_xscale("log")
            ax.set_title(f"{o} forecast")
            ax.set_xlabel("credit per neighbour / credit per off-axis lane (log)")
        axes[0].set_yticks(range(len(arms)), arms, fontsize=7)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.suptitle("Exp 2 SHAP, all-lanes models: which other lanes a forecast leans on")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(figdir / "lane_affinity.png", dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data-dir", default=".")
    r.add_argument("--data-csv", default="data/processed/hs2_monthly_aggregated.csv")
    r.add_argument("--train-start", default="2009-01-01")
    r.add_argument("--train-end", default="2023-12-01")
    r.add_argument("--test-start", default="2025-01-01")
    r.add_argument("--test-end", default="2025-05-01")
    r.add_argument("--lanes-per-tier", type=int, default=8)
    r.add_argument("--all-lanes", action="store_true", help="explain every kept lane")
    r.add_argument("--lanes-file", default=None,
                   help="CSV of State,Commodity,Import/Export lanes to explain")
    r.add_argument("--kinds", nargs="+", default=["baseline", "flat", "tree", "combo"])
    r.add_argument("--combo-seeds", nargs="*", type=int, default=[26])
    r.add_argument("--seeds", nargs="*", type=int, default=None, help="only these seeds, every kind")
    r.add_argument("--bg-flat", type=int, default=32)
    r.add_argument("--steps-flat", type=int, default=16)
    r.add_argument("--bg-combo", type=int, default=4)
    r.add_argument("--steps-combo", type=int, default=8)
    r.add_argument("--device", default="cuda")
    r.add_argument("--only", nargs="*", default=None)
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--skip-models", nargs="*", default=[],
                   help="e.g. random_forest, whose CPU TreeSHAP can run as a separate job")
    r.add_argument("--rf-workers", type=int, default=48, help="CPU workers for forest TreeSHAP")
    s = sub.add_parser("summarise")
    s.add_argument("--out", required=True)
    args = p.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_summarise(args)


if __name__ == "__main__":
    main()
