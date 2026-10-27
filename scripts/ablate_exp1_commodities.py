#!/usr/bin/env python
"""Commodity ablation for Exp 1 (one aggregate series), inference only.

Exp 1 models see the eight-state aggregate, which is the sum of every lane
(all 1,542 state x commodity x flow series, before any sparse-lane filter).
This script asks how much each large commodity moves those forecasts:

  1. rank HS2 commodities by training-period value and by training tonnage
     and take the top N of each (individually, plus each top-N block);
  2. replace that commodity's lanes with their own training-period mean,
     for value only, tonnage only, or both, BEFORE the lanes are summed;
  3. rebuild the aggregate, its scaling (the ORIGINAL training min/max, so
     only the ablated history changes), lags and windows exactly as the
     Exp 1 pipeline does;
  4. run the original trained checkpoint (no retraining) and compare its
     forecasts with the unablated ones on the 12 validation and 5 test months.

The training mean (not zero) keeps an ablated aggregate inside the range the
model was trained on: removing a large commodity outright would mostly test
the model on an out-of-range level.

Each run is gated: its unablated test forecasts must match predictions.csv.
Read-only on --runs-dir.

Usage:
    python scripts/ablate_exp1_commodities.py --runs-dir outputs/agg --out outputs/ablation_exp1
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
    BASELINES, PRED_TOL, TREES, checkpoint_for, construct_model, discover_runs,
    load_torch_weights, resolve_cfg, saved_test_predictions,
)
from src.data import DataConfig, TradeDataPipeline  # noqa: E402
from src.data.dataloader import KEYS, VALUE_CANON, WEIGHT_CANON  # noqa: E402
from src.utils import get_data_config, set_seed  # noqa: E402

PARTS = {"value": [VALUE_CANON], "weight": [WEIGHT_CANON], "both": [VALUE_CANON, WEIGHT_CANON]}
OUTPUTS = ("value", "weight")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--input-len", type=int, default=36)
    p.add_argument("--lag-count", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=33)
    p.add_argument("--ma-window", type=int, default=12)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def build(cfg, seed, ablation=None, stats=None):
    """Exp 1 pipeline; with ``ablation`` the lane panel is edited before the
    aggregate is summed and the original training stats are reused."""
    set_seed(seed)
    pipe = TradeDataPipeline(DataConfig(**get_data_config(cfg)))
    pipe.load_data()
    if ablation is not None:
        original_build = pipe.build_grid
        pipe.build_grid = lambda df: ablate_grid(original_build(df), ablation, pipe.config)
        pipe.compute_stats = lambda df: stats
    pipe.create_splits()
    loaders = pipe.get_dataloaders(batch_size=cfg["training"]["batch_size"], num_workers=0,
                                   shuffle_train=False, combo=False)
    return pipe, loaders


def ablate_grid(grid, ablation, config):
    commodities, part = ablation
    grid = grid.copy()
    mask = grid["Commodity"].isin(commodities)
    train = mask & (grid["Time"] <= pd.Timestamp(config.train_end))
    means = grid[train].groupby(KEYS)[PARTS[part]].mean()
    idx = pd.MultiIndex.from_frame(grid.loc[mask, KEYS])
    for col in PARTS[part]:
        grid.loc[mask, col] = means[col].reindex(idx).to_numpy()
    if grid.loc[mask, PARTS[part]].isna().any().any():
        raise ValueError(f"lanes without training data in {commodities}")
    return grid


def commodity_ranking(pipe, top):
    grid = pipe.build_grid(pipe.df_full)
    train = grid[grid["Time"] <= pd.Timestamp(pipe.config.train_end)]
    tot = train.groupby("Commodity")[[VALUE_CANON, WEIGHT_CANON]].sum()
    share = tot / tot.sum()
    rank = pd.DataFrame({
        "train_value_usd": tot[VALUE_CANON], "train_tonnage_kg": tot[WEIGHT_CANON],
        "value_share": share[VALUE_CANON], "tonnage_share": share[WEIGHT_CANON],
        "value_rank": tot[VALUE_CANON].rank(ascending=False, method="first").astype(int),
        "tonnage_rank": tot[WEIGHT_CANON].rank(ascending=False, method="first").astype(int),
    }).sort_values("value_rank")
    top_value = rank.index[rank.value_rank <= top].tolist()
    top_tonnage = rank.index[rank.tonnage_rank <= top].tolist()
    # CONTROL ablates nothing: it must reproduce the unablated forecasts
    # exactly, which checks that the rebuilt pipeline matches Exp 1.
    groups = {"CONTROL": []}
    groups.update({c: [c] for c in dict.fromkeys(top_value + top_tonnage)})
    groups[f"TOP{top}_BY_VALUE"] = top_value
    groups[f"TOP{top}_BY_TONNAGE"] = top_tonnage
    return rank.reset_index(), groups


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def windows(loaders):
    """Validation + test samples in time order: x, ids, times, targets."""
    va, te = loaders[1].dataset.samples, loaders[2].dataset.samples
    samples = list(va) + list(te)
    x = torch.stack([s["x_numeric"] for s in samples])
    ids = [torch.tensor([s[k] for s in samples]) for k in ("state_id", "comm_id", "flow_id")]
    times = [pd.Timestamp(s["target_time"]) for s in samples]
    y = np.array([[float(s["target_value"]), float(s["target_weight"])] for s in samples])
    split = ["val"] * len(va) + ["test"] * len(te)
    return x, ids, times, y, split


def predict(net, model, loaders, device):
    if model in TREES:
        pv, _ = net.predict_loader(loaders[1])
        pt, _ = net.predict_loader(loaders[2])
        return np.vstack([pv, pt]).astype(np.float64)
    x, ids, *_ = windows(loaders)
    with torch.no_grad():
        out = net(x.to(device), *(t.to(device) for t in ids))
    return out.double().cpu().numpy()


def real_units(norm, stats):
    row = stats["exact"].iloc[0]
    lo = np.array([row["val_min"], row["wt_min"]], dtype=np.float64)
    span = np.array([row["val_max"] - row["val_min"], row["wt_max"] - row["wt_min"]], dtype=np.float64)
    span[span == 0] = 1.0
    return norm * span + lo


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    runs = discover_runs(args.runs_dir, args.only)
    print(f"{len(runs)} runs on {device}", flush=True)

    base_cache, abl_cache = {}, {}
    rows, checks, ranking, groups = [], [], None, None
    for run in sorted(runs, key=lambda r: (r["model"], r["variant"], r["seed"])):
        rec = {k: run[k] for k in ("run", "model", "variant", "seed")}
        t0 = time.time()
        try:
            model = run["model"]
            cfg = resolve_cfg(args, model, run["variant"], checkpoint_for(run))
            key = json.dumps(get_data_config(cfg), sort_keys=True, default=str)
            if key not in base_cache:
                base_cache[key] = build(cfg, run["seed"])
            pipe, loaders = base_cache[key]
            if ranking is None:
                ranking, groups = commodity_ranking(pipe, args.top)
                ranking.to_csv(out / "commodity_ranking.csv", index=False)
                print(f"{len(groups)} ablation groups x {len(PARTS)} parts", flush=True)

            net = construct_model(cfg, pipe, model, run["variant"], run["seed"], args.ma_window)
            if model in TREES:
                net.load(str(checkpoint_for(run)))
            else:
                if model not in BASELINES:
                    load_torch_weights(net, checkpoint_for(run), device)
                net = net.to(device).eval()

            _, _, times, y_norm, split = windows(loaders)
            base = predict(net, model, loaders, device)
            saved, _ = saved_test_predictions(run["dir"])
            test = np.array(split) == "test"
            rec["max_pred_diff"] = float(np.abs(base[test] - saved).max())
            if rec["max_pred_diff"] > PRED_TOL:
                raise RuntimeError(f"unablated forecasts differ from predictions.csv by {rec['max_pred_diff']:.2e}")
            base_real = real_units(base, pipe.stats)
            y_real = real_units(y_norm, pipe.stats)

            for gname, commodities in groups.items():
                for part in PARTS:
                    akey = (key, gname, part)
                    if akey not in abl_cache:
                        abl_cache[akey] = build(cfg, run["seed"], (commodities, part), pipe.stats)
                    apipe, aloaders = abl_cache[akey]
                    pred = predict(net, model, aloaders, device)
                    if gname == "CONTROL":
                        rec[f"control_diff_{part}"] = float(np.abs(pred - base).max())
                        if rec[f"control_diff_{part}"] > 1e-6:
                            raise RuntimeError(f"control ablation changed forecasts by {rec[f'control_diff_{part}']:.2e}")
                    pred_real = real_units(pred, pipe.stats)
                    for i, t in enumerate(times):
                        for o, name in enumerate(OUTPUTS):
                            rows.append({
                                "run": run["run"], "model": model, "variant": run["variant"],
                                "seed": run["seed"], "group": gname, "part": part,
                                "split": split[i], "time": t.date().isoformat(), "output": name,
                                "pred_norm": base[i, o], "ablated_norm": pred[i, o],
                                "pred_real": base_real[i, o], "ablated_real": pred_real[i, o],
                                "actual_real": y_real[i, o],
                            })
            rec["status"] = "ok"
        except Exception as exc:  # one broken run must not stop the sweep
            rec.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        rec["seconds"] = round(time.time() - t0, 1)
        checks.append(rec)
        print(f"  {run['run']}: {rec['status']} pred_diff={rec.get('max_pred_diff', np.nan):.1e} "
              f"{rec['seconds']}s", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    long = pd.DataFrame(rows)
    long.to_csv(out / "ablation_long.csv.gz", index=False)
    summarise(out, long, pd.DataFrame(checks), ranking, groups, args.top)


def summarise(out, long, checks, ranking, groups, top):
    d = long.copy()
    d["delta_real"] = d.ablated_real - d.pred_real
    d["delta_pct"] = 100 * d.delta_real / d.pred_real.abs()
    d["abs_err"] = (d.pred_real - d.actual_real).abs()
    d["abs_err_ablated"] = (d.ablated_real - d.actual_real).abs()
    test = d[d.split == "test"]
    per_run = (test.groupby(["model", "variant", "seed", "group", "part", "output"])
               .agg(mean_delta_pct=("delta_pct", "mean"), mean_abs_delta_pct=("delta_pct", lambda s: s.abs().mean()),
                    mae=("abs_err", "mean"), mae_ablated=("abs_err_ablated", "mean"))
               .reset_index())
    per_run["mae_change_pct"] = 100 * (per_run.mae_ablated - per_run.mae) / per_run.mae
    by_arm = (per_run.groupby(["model", "variant", "group", "part", "output"])
              .agg(seeds=("seed", "nunique"), mean_delta_pct=("mean_delta_pct", "mean"),
                   mean_abs_delta_pct=("mean_abs_delta_pct", "mean"),
                   sd_abs_delta_pct=("mean_abs_delta_pct", "std"),
                   mae_change_pct=("mae_change_pct", "mean"))
              .reset_index())
    learned = by_arm[~by_arm.model.isin(BASELINES)]
    by_group = (learned.groupby(["group", "part", "output"])
                .agg(arms=("model", "size"), median_abs_delta_pct=("mean_abs_delta_pct", "median"),
                     max_abs_delta_pct=("mean_abs_delta_pct", "max"),
                     median_mae_change_pct=("mae_change_pct", "median"))
                .reset_index())
    # Additivity: sum of single-commodity effects vs the matching top-N block.
    add = []
    singles = {g for g, c in groups.items() if len(c) == 1}  # CONTROL has none
    for block in (f"TOP{top}_BY_VALUE", f"TOP{top}_BY_TONNAGE"):
        members = [c for c in groups[block] if c in singles]
        s = (per_run[per_run.group.isin(members)]
             .groupby(["model", "variant", "seed", "part", "output"]).mean_delta_pct.sum().rename("sum_of_singles"))
        b = (per_run[per_run.group == block]
             .set_index(["model", "variant", "seed", "part", "output"]).mean_delta_pct.rename("block"))
        j = pd.concat([s, b], axis=1).reset_index()
        j["block_group"] = block
        add.append(j)
    add = pd.concat(add).groupby(["block_group", "model", "variant", "part", "output"])[
        ["sum_of_singles", "block"]].mean().reset_index()
    notes = pd.DataFrame({"note": [
        "Exp 1 commodity ablation, inference only: original trained checkpoints, no retraining.",
        "An ablated commodity's lanes (all states and flows) are replaced by each lane's own training-period mean",
        "(2009-2023) for value only, tonnage only, or both, before the lanes are summed into the aggregate.",
        "Scaling uses the ORIGINAL training min/max; lags and windows are rebuilt exactly as in Exp 1.",
        "delta_pct = (ablated - original forecast) / |original forecast| in real units (US$ / kg), test months.",
        "mae_change_pct > 0: the ablated history makes the test forecast worse against the actual outcome.",
        f"Groups: the top {top} HS2 commodities by training value and by training tonnage (each alone),",
        f"plus each top-{top} block. Commodity_Ranking lists every commodity's training share.",
        "Gate: each run's unablated test forecasts must match its predictions.csv within 1e-4 (normalized).",
        "This measures how much each commodity's history moves a fixed model's forecast; Exp 1 models never",
        "see commodities, so it is not a learned commodity effect.",
    ]})
    with pd.ExcelWriter(out / "ablation_exp1.xlsx") as xw:
        notes.to_excel(xw, sheet_name="Notes", index=False)
        checks.to_excel(xw, sheet_name="Checks", index=False)
        ranking.to_excel(xw, sheet_name="Commodity_Ranking", index=False)
        by_group.to_excel(xw, sheet_name="Effect_by_group", index=False)
        by_arm.to_excel(xw, sheet_name="Effect_by_arm", index=False)
        add.to_excel(xw, sheet_name="Additivity", index=False)
        per_run.to_excel(xw, sheet_name="Effect_by_run", index=False)
    print(f"Wrote {out / 'ablation_exp1.xlsx'}; {int((checks.status == 'ok').sum())}/{len(checks)} runs ok")
    try:
        plot(out, by_arm, groups, top)
    except Exception:
        traceback.print_exc()


def plot(out, by_arm, groups, top):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    singles = [g for g, c in groups.items() if len(c) == 1] + [f"TOP{top}_BY_VALUE", f"TOP{top}_BY_TONNAGE"]  # no CONTROL
    d = by_arm.copy()
    d["arm"] = d.model + " / " + d.variant
    arms = sorted(d.arm.unique())
    for part in PARTS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 0.36 * len(arms) + 3), sharey=True)
        for ax, o in zip(axes, OUTPUTS):
            m = (d[(d.part == part) & (d.output == o)]
                 .pivot_table(index="arm", columns="group", values="mean_delta_pct").reindex(index=arms, columns=singles))
            # Symmetric log scale: HS 27 moves forecasts by ~45% while most
            # commodities move them by well under 1%, so a linear scale hides them.
            from matplotlib.colors import SymLogNorm
            lim = float(np.nanmax(np.abs(m.to_numpy()))) or 1.0
            norm = SymLogNorm(linthresh=0.05, linscale=0.5, vmin=-lim, vmax=lim, base=10)
            im = ax.imshow(m.to_numpy(), aspect="auto", cmap="RdBu_r", norm=norm)
            ax.set_xticks(range(len(singles)), [s[:22] for s in singles], rotation=70, ha="right", fontsize=7)
            ax.set_title(f"{o} forecast")
            fig.colorbar(im, ax=ax, label="mean forecast change, % (test months; symmetric log scale)")
        axes[0].set_yticks(range(len(arms)), arms, fontsize=7)
        fig.suptitle(f"Exp 1: replace a commodity's {part if part != 'both' else 'value and tonnage'} "
                     "history with its training mean")
        fig.tight_layout()
        fig.savefig(figdir / f"ablation_{part}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
