#!/usr/bin/env python
"""SHAP attributions for the Exp 2 arms trained through external/s4.

Companion to scripts/explain_exp2.py, which covers every native arm and owns
the lane sample (lanes.csv), the workbook and the figures. This script writes
the same per-run .npz / checks.jsonl records into the same --out directory,
so ``explain_exp2.py summarise`` picks the S4 arms up with the others.

It is separate because external/s4 has its own top-level ``src`` package and
must be imported the way scripts/s4_eval.py imports it; nothing from the
repository's ``src`` is imported here.

Arms and inputs:
  * s4_s4_{embeddings,onehot}[_bidir]: one lane per sample, (L, F) history
    (one-hot ID columns, when present, are held fixed and not attributed);
    baselines are the lane's own training windows.
  * s4_s4nd_{2d,2d_state,3d,4d}: one sample is a grid of lanes x (L, F); a
    lane's forecast can draw on every lane in its grid, so credit is kept per
    source lane. Baselines are training samples of the same grid slot.

Run from the repository root:
    python scripts/explain_exp2_s4.py --runs-dir outputs/full --out outputs/shap_exp2
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/s4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

PRED_TOL = 1e-4
COMPLETENESS_TOL = 0.02
FLOW = ["Export", "Import"]
# Integration points per forward pass; the full 4-D grid does not fit eight.
GRID_CHUNK = {"s4nd_4d": 2}


def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


def channel_index(feat_cols):
    """0 value, 1 weight, 2 season per feature column (same rule as explain_exp1)."""
    out = []
    for j, col in enumerate(feat_cols):
        name = str(col).lower()
        if name.startswith("value_lag_") or (j == 0 and not name.startswith(("weight", "sin", "cos"))):
            out.append(0)
        elif name.startswith("weight_lag_") or j == 1:
            out.append(1)
        elif name in ("sin_month", "cos_month"):
            out.append(2)
        else:
            raise ValueError(f"unmapped feature column {col!r}")
    return np.array(out)


def load_module(run_dir, exp, seed, device):
    """Rebuild the Lightning module exactly as scripts/s4_eval.py evaluate() does."""
    import src.utils.train  # noqa: F401  (registers process_config)
    from train import SequenceLightningModule
    ckpt = run_dir / "hydra/checkpoints/val/loss.ckpt"
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["benchmark_config"])
    if int(config.train.seed) != seed:
        raise RuntimeError("checkpoint seed does not match run directory")
    if os.environ.get("DATA_DIR"):
        config.dataset.data_dir = os.path.abspath(os.environ["DATA_DIR"])
    config.wandb = None
    module = SequenceLightningModule(config)
    result = module.load_state_dict(checkpoint["state_dict"], strict=False)
    miss = [k for k in result.missing_keys if not k.endswith(("kernel.L", "kernel.w"))]
    if miss or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={miss}, unexpected={result.unexpected_keys}")
    module.eval().to(device)
    module._initialize_state()
    return module, config


def exp_name(run_name):
    """s4_s4nd_3d_seed26 -> s4nd_3d ; s4_s4_onehot_bidir_seed105 -> s4_onehot_bidir."""
    return run_name[len("s4_"):].rsplit("_seed", 1)[0]


def to_device(z, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in z.items()}


def expand_z(z, n):
    return {k: (v.expand(n, *v.shape[1:]) if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1 else v)
            for k, v in z.items()}


# --------------------------------------------------------------------------- #
# Per-series (1-D S4)
# --------------------------------------------------------------------------- #
def explain_series(module, lanes, times, device, bg_n, steps):
    ds = module.dataset
    pipe = ds.pipe
    id2s = {v: k for k, v in pipe.state2id.items()}
    id2c = {v: k for k, v in pipe.comm2id.items()}
    id2f = {v: k for k, v in pipe.flow2id.items()}

    def by_lane(split):
        out = {}
        dset = ds._ds[split]
        for i, s in enumerate(dset.samples):
            key = (str(id2s[s["state_id"]]), str(id2c[s["comm_id"]]), str(id2f[s["flow_id"]]))
            out.setdefault(key, []).append(i)
        return dset, out

    tr_ds, tr_idx = by_lane("train")
    te_ds, te_idx = by_lane("test")
    F = len(pipe.feat_cols)
    L = int(pipe.config.input_len)
    nodes, weights = gauss_legendre(steps)
    a = torch.tensor(nodes, dtype=torch.float32, device=device)
    w = torch.tensor(weights, dtype=torch.float64, device=device)
    own = np.zeros((len(lanes), len(times), 2, L, F))
    fx = np.zeros((len(lanes), len(times), 2))
    base = np.zeros_like(fx)
    for li, key in enumerate(lanes):
        te_items = {pd.Timestamp(te_ds.samples[i]["target_time"]): i for i in te_idx[key]}
        x, _, z = ds._collate_fn([te_ds[te_items[t]] for t in times])
        tr = tr_idx[key]
        pick = [tr[int(i)] for i in np.unique(np.linspace(0, len(tr) - 1, min(bg_n, len(tr))).round())]
        bgx, _, _ = ds._collate_fn([tr_ds[i] for i in pick])
        x, bgx = x.to(device), bgx.to(device)
        z = to_device(z, device)
        extra = x[..., F:]                                          # one-hot IDs (held fixed)

        def f(num):
            n = num.shape[0]
            full = torch.cat([num, extra[:1].expand(n, -1, -1)], -1) if extra.shape[-1] else num
            zz = {k: (v[:1].expand(n, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in z.items()}
            out, _, _ = module.forward((full, torch.zeros(n, 2, device=device), zz))
            return out.reshape(n, 2)

        xn, bn = x[..., :F], bgx[..., :F]
        with torch.no_grad():
            fx[li] = f(xn).double().cpu().numpy()
            base[li] = f(bn).double().mean(0).cpu().numpy()
        diff = xn[:, None] - bn[None]
        pts = (bn[None, :, None] + a.view(1, 1, -1, 1, 1) * diff[:, :, None]).reshape(-1, L, F)
        pts = pts.detach().requires_grad_(True)
        out = f(pts)
        for j in range(2):
            (g,) = torch.autograd.grad(out[:, j].sum(), pts, retain_graph=j == 0)
            g = g.view(len(times), len(pick), steps, L, F).double()
            own[li, :, j] = ((g * w.view(1, 1, -1, 1, 1)).sum(2) * diff.double()).mean(1).cpu().numpy()
    return {"own": own, "fx": fx, "base": base}


# --------------------------------------------------------------------------- #
# Grid (S4ND)
# --------------------------------------------------------------------------- #
def grid_layout(ds, exp):
    """Per-sample slot and per-cell lane key for each S4ND variant."""
    if exp == "s4nd_3d":
        S, C = ds.n_states, ds.n_comms
        slots = lambda n_months: [(m, fid) for m in range(n_months) for fid in (0, 1)]  # noqa: E731
        cell_key = lambda slot: {(s, c): (ds.states[s], ds.comms[c], FLOW[slot[1]])  # noqa: E731
                                 for s in range(S) for c in range(C)}
    elif exp == "s4nd_4d":
        S, C = len(ds.states), len(ds.comms)
        slots = lambda n_months: [(m, None) for m in range(n_months)]  # noqa: E731
        cell_key = lambda slot: {(s, c, f): (ds.states[s], ds.comms[c], FLOW[f])  # noqa: E731
                                 for s in range(S) for c in range(C) for f in (0, 1)}
    elif exp == "s4nd_2d":
        S, G = len(ds.states), len(ds.groups)
        slots = lambda n_months: [(m, None) for m in range(n_months)]  # noqa: E731
        cell_key = lambda slot: {(s, g): (ds.states[s], ds.groups[g][0], ds.groups[g][1])  # noqa: E731
                                 for s in range(S) for g in range(G)}
    elif exp == "s4nd_2d_state":
        S, G = len(ds.states), len(ds.groups)
        slots = lambda n_months: [(m, g) for m in range(n_months) for g in range(G)]  # noqa: E731
        cell_key = lambda slot: {(s,): (ds.states[s], ds.groups[slot[1]][0], ds.groups[slot[1]][1])  # noqa: E731
                                 for s in range(S)}
    else:
        raise ValueError(exp)
    return slots, cell_key


def collate_one(ds, dataset, i):
    """One sample through the dataset's own loader, so extras become the z dict."""
    from torch.utils.data import Subset
    return next(iter(ds._dataloader(Subset(dataset, [i]), batch_size=1, shuffle=False)))


def explain_grid(module, exp, lanes, times, combo_keys, device, bg_n, steps):
    ds = module.dataset
    L = int(ds.pipe.config.input_len)
    chan = channel_index(ds.pipe.feat_cols)
    slots, cell_key = grid_layout(ds, exp)
    n_train = len(ds.dataset_train)
    n_test = len(ds.dataset_test)
    per = len(slots(1))
    te_slots = slots(n_test // per)
    tr_slots = slots(n_train // per)
    if len(te_slots) != n_test or len(tr_slots) != n_train:
        raise RuntimeError("grid sample count does not match the expected slot layout")
    test_months = sorted({pd.Timestamp(t) for t in ds.test_times})
    if [pd.Timestamp(t) for t in times] != test_months:
        raise RuntimeError("explained months differ from the S4ND test months")
    # Locate each target lane: (test sample index, cell index).
    where = {}
    for i, slot in enumerate(te_slots):
        for cell, key in cell_key(slot).items():
            where[(tuple(map(str, key)), slot[0])] = (i, cell)
    g_of = {k: g for g, k in enumerate(combo_keys)}
    nodes, weights = gauss_legendre(steps)
    own = np.zeros((len(lanes), len(times), 2, L, len(chan)))
    src = np.zeros((len(lanes), len(times), 2, len(combo_keys), 3))
    fx = np.zeros((len(lanes), len(times), 2))
    base = np.zeros_like(fx)
    loader_kw = dict(batch_size=1, shuffle=False)
    test_batches = list(ds._dataloader(ds.dataset_test, **loader_kw))
    train_batches = None
    for li, key in enumerate(lanes):
        for ti in range(len(times)):
            i, cell = where[(key, ti)]
            x, _, z = test_batches[i]
            slot = te_slots[i]
            same = [k for k, s in enumerate(tr_slots) if s[1] == slot[1]]
            pick = [same[int(p)] for p in np.unique(np.linspace(0, len(same) - 1, min(bg_n, len(same))).round())]
            if train_batches is None:
                train_batches = {}
            for p in pick:
                if p not in train_batches:
                    train_batches[p] = collate_one(ds, ds.dataset_train, p)
            bg = torch.cat([train_batches[p][0] for p in pick]).to(device)          # (B, *cells, L, F)
            x = x.to(device)
            z = to_device(z, device)

            def f(p):
                zz = expand_z(z, p.shape[0])
                out, _, _ = module.forward((p, torch.zeros(1, device=device), zz))
                return out[(slice(None),) + tuple(cell)]                            # (n, 2)

            with torch.no_grad():
                fx[li, ti] = f(x).double().cpu().numpy()[0]
                base[li, ti] = f(bg).double().mean(0).cpu().numpy()
            acc = torch.zeros((2,) + tuple(x.shape[1:]), dtype=torch.float64, device=device)
            shape = (-1,) + (1,) * (x.dim() - 1)
            chunk = GRID_CHUNK.get(exp, len(nodes))
            for b in range(bg.shape[0]):
                diff = x - bg[b:b + 1]
                for k0 in range(0, len(nodes), chunk):
                    ak = torch.tensor(nodes[k0:k0 + chunk], dtype=torch.float32, device=device).view(shape)
                    wk = torch.tensor(weights[k0:k0 + chunk], dtype=torch.float64, device=device).view(shape)
                    pts = (bg[b:b + 1] + ak * diff).detach().requires_grad_(True)
                    out = f(pts)
                    for j in range(2):
                        (g,) = torch.autograd.grad(out[:, j].sum(), pts, retain_graph=j == 0)
                        acc[j] += (g.double() * wk * diff.double()).sum(0) / bg.shape[0]
                    del out, pts
            a = acc.cpu().numpy()                                                   # (2, *cells, L, F)
            own[li, ti] = a[(slice(None),) + tuple(cell)]
            for c_idx, lane in cell_key(slot).items():
                lane = tuple(map(str, lane))
                if lane == key or lane not in g_of:
                    continue
                v = a[(slice(None),) + tuple(c_idx)]                                # (2, L, F)
                for c in range(3):
                    src[li, ti, :, g_of[lane], c] = v[..., chan == c].sum((1, 2))
    return {"own": own, "src": src, "fx": fx, "base": base,
            "combo_keys": np.array(["|".join(k) for k in combo_keys])}


# --------------------------------------------------------------------------- #
def gates(res, run_dir, lanes, times):
    saved = saved_predictions(run_dir, lanes, times)
    pred_diff = float(np.nanmax(np.abs(res["fx"] - saved))) if np.isfinite(saved).all() else np.inf
    total = res["own"].sum((-2, -1)) + (res["src"].sum((-2, -1)) if "src" in res else 0)
    resid = total - (res["fx"] - res["base"])
    scale = np.maximum(np.abs(res["fx"] - res["base"]).reshape(-1, 2).mean(0), 1e-6)
    return pred_diff, float((np.abs(resid).reshape(-1, 2) / scale).max())


def saved_predictions(run_dir, lanes, times):
    p = run_dir / "predictions_norm.csv"
    df = pd.read_csv(p, parse_dates=["Time"])
    idx = {(str(s), str(c), str(f), pd.Timestamp(t)): (v, w) for s, c, f, t, v, w in zip(
        df["State"], df["Commodity"], df["Import/Export"], df["Time"], df["pred_value"], df["pred_weight"])}
    out = np.full((len(lanes), len(times), 2), np.nan)
    for i, k in enumerate(lanes):
        for j, t in enumerate(times):
            if (*k, t) in idx:
                out[i, j] = idx[(*k, t)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out", required=True, help="same directory as explain_exp2.py run")
    p.add_argument("--seeds", nargs="*", type=int, default=[26, 105, 115],
                   help="seeds for the per-series S4 arms (cheap)")
    p.add_argument("--grid-seeds", nargs="*", type=int, default=[26],
                   help="seeds for the S4ND grid arms (expensive)")
    p.add_argument("--bg-flat", type=int, default=32)
    p.add_argument("--steps-flat", type=int, default=16)
    p.add_argument("--bg-grid", type=int, default=4)
    p.add_argument("--steps-grid", type=int, default=8)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    out = Path(args.out).resolve()
    lanes_df = pd.read_csv(out / "lanes.csv")
    lanes = [(str(r["State"]), str(r["Commodity"]), str(r["Import/Export"])) for _, r in lanes_df.iterrows()]
    times = [pd.Timestamp(t) for t in pd.date_range("2025-01-01", "2025-05-01", freq="MS")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(ROOT / "external/s4")          # hydra-relative paths, as in run_full_test.sh
    for mp in sorted(runs_dir.glob("s4_s4*/metrics.json")):
        m = json.loads(mp.read_text())
        run, seed = mp.parent.name, int(m["seed"])
        grid = exp_name(run).startswith("s4nd")
        if seed not in (args.grid_seeds if grid else args.seeds):
            continue
        if args.only and run not in args.only:
            continue
        dest = out / "per_run" / f"{run}.npz"
        if dest.exists() and not args.overwrite and str(np.load(dest)["status"]) == "ok":
            print(f"  {run}: done, skipped", flush=True)
            continue
        exp = exp_name(run)
        kind = "hydra_s4nd" if exp.startswith("s4nd") else "hydra_s4"
        rec = {"run": run, "model": m["model"], "variant": m["variant"], "seed": seed, "kind": kind}
        t0 = time.time()
        try:
            module, _ = load_module(mp.parent, exp, seed, device)
            pipe = module.dataset.pipe
            combo_keys = [tuple(map(str, k)) for k in pipe.kept_combos]
            def attribute(scale):
                if kind == "hydra_s4":
                    return explain_series(module, lanes, times, device, args.bg_flat,
                                          scale * args.steps_flat)
                return explain_grid(module, exp, lanes, times, combo_keys, device,
                                    args.bg_grid, scale * args.steps_grid)

            res = attribute(1)
            for factor in (2, 4):                          # quadrature only: more nodes
                pred_diff, comp = gates(res, mp.parent, lanes, times)
                if comp <= COMPLETENESS_TOL or pred_diff > PRED_TOL:
                    break
                rec.setdefault("retries", []).append({"factor": factor // 2, "gap": comp})
                res = attribute(factor)
            saved = saved_predictions(mp.parent, lanes, times)
            pred_diff = float(np.nanmax(np.abs(res["fx"] - saved))) if np.isfinite(saved).all() else np.inf
            total = res["own"].sum((-2, -1)) + (res["src"].sum((-2, -1)) if "src" in res else 0)
            resid = total - (res["fx"] - res["base"])
            scale = np.maximum(np.abs(res["fx"] - res["base"]).reshape(-1, 2).mean(0), 1e-6)
            comp = float((np.abs(resid).reshape(-1, 2) / scale).max())
            ok = pred_diff <= PRED_TOL and comp <= COMPLETENESS_TOL
            rec.update(status="ok" if ok else "FAILED", max_pred_diff=pred_diff,
                       max_completeness_gap=comp, max_abs_residual=float(np.abs(resid).max()),
                       n_lanes=len(lanes), n_months=len(times))
            extra = {k: v for k, v in res.items() if k not in ("own", "fx", "base")}
            np.savez_compressed(dest, own=res["own"].astype(np.float32), fx=res["fx"], base=res["base"],
                                lanes=np.array(["|".join(k) for k in lanes]),
                                times=np.array([str(t.date()) for t in times]),
                                feat_cols=np.array(pipe.feat_cols), status=rec["status"],
                                **{k: (v.astype(np.float32) if k == "src" else v) for k, v in extra.items()})
        except Exception as exc:
            rec.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        rec["seconds"] = round(time.time() - t0, 1)
        with open(out / "checks.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"  {run}: {rec.get('status')} pred_diff={rec.get('max_pred_diff', np.nan):.2e} "
              f"gap={rec.get('max_completeness_gap', np.nan):.2e} {rec['seconds']}s", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
