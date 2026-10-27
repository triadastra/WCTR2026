"""Independent recomputation of every paper number from raw run files.

Usage (from the repository root):
  python tools/recompute_results.py [SNAPSHOT_DIR DATASET_CSV OUT_DIR]
Defaults: results/2026-10-26  data/processed/hs2_monthly_aggregated.csv
          docs/paper_audit/recomputed
Writes OUT_DIR/computed.json plus per-table CSVs. Uses only metrics.json,
predictions.csv, the SHAP arrays and the raw monthly dataset; summaries and
analysis CSVs are NOT read here (they are compared later as a cross-check).
Needs numpy, pandas and scipy; runs on CPU in about a minute.
"""
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_ARGS = sys.argv[1:4] or [ROOT / "results/2026-10-26", ROOT / "data/processed/hs2_monthly_aggregated.csv",
                          ROOT / "docs/paper_audit/recomputed"]
SNAP, DATA, OUT = (Path(a) for a in _ARGS)
OUT.mkdir(parents=True, exist_ok=True)
KEY = ["State", "Commodity", "Import/Export", "Time"]
RESULT = {}


def smape(t, p):
    denom = np.abs(t) + np.abs(p)
    terms = np.zeros_like(denom)
    ok = denom > 1e-12
    terms[ok] = 2.0 * np.abs(p[ok] - t[ok]) / denom[ok]
    return terms.mean() * 100


def recompute(df):
    ev = (df.pred_value - df.true_value).to_numpy(float)
    ew = (df.pred_weight - df.true_weight).to_numpy(float)
    return {
        "mae": (np.abs(ev).mean() + np.abs(ew).mean()) / 2,
        "mse": ((ev ** 2).mean() + (ew ** 2).mean()) / 2,
        "rmse": (np.sqrt((ev ** 2).mean()) + np.sqrt((ew ** 2).mean())) / 2,
        "smape": (smape(df.true_value_real.to_numpy(float), df.pred_value_real.to_numpy(float))
                  + smape(df.true_weight_real.to_numpy(float), df.pred_weight_real.to_numpy(float))) / 2,
    }


def get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


RAW = pd.read_csv(DATA, parse_dates=["Time"])
_TRN = RAW[(RAW.Time >= "2009-01-01") & (RAW.Time <= "2023-12-01")]
MINMAX = _TRN.groupby(["State", "Commodity", "Import/Export"]).agg(
    vmin=("Value", "min"), vmax=("Value", "max"), wmin=("Weight", "min"), wmax=("Weight", "max"))
SUBSTITUTED = []   # runs whose declared predictions file is absent or real-unit only


def normalize_real(df):
    """Rebuild normalized columns from real units with train-period min/max per lane."""
    m = df.merge(MINMAX, left_on=["State", "Commodity", "Import/Export"], right_index=True, how="left")
    out = df.copy()
    for col, lo, hi in (("value", "vmin", "vmax"), ("weight", "wmin", "wmax")):
        rng = (m[hi] - m[lo]).to_numpy(float)
        out[f"pred_{col}"] = (m[f"pred_{col}_real"] - m[lo]).to_numpy(float) / rng
        out[f"true_{col}"] = (m[f"true_{col}_real"] - m[lo]).to_numpy(float) / rng
    return out


def load(sub):
    rows, preds = [], {}
    for mj in sorted((SNAP / "runs" / sub).glob("*/metrics.json")):
        d = json.loads(mj.read_text(encoding="utf-8"))
        declared = mj.parent / d.get("predictions_file", "predictions.csv")
        df = pd.read_csv(declared if declared.exists() else mj.parent / "predictions.csv")
        if not declared.exists() or np.allclose(df.true_value, df.true_value_real):
            if np.allclose(df.true_value, df.true_value_real):
                df = normalize_real(df)
            SUBSTITUTED.append({"run": f"{sub}/{mj.parent.name}", "declared": declared.name,
                                "declared_exists": declared.exists(), "renormalized": True})
        rc = d.get("run_contract", {})
        m = d.get("metrics", {})
        row = {"run": mj.parent.name, "model": d["model"], "variant": d["variant"], "seed": d["seed"],
               "arm": f"{d['model']}|{d['variant']}", "n": len(df),
               "rep_mae": m.get("mae"), "rep_mse": m.get("norm_mse_combined"), "rep_rmse": m.get("rmse"),
               "rep_smape": m.get("smape"), "rep_mase": m.get("mase"),
               "input_len": get(rc, "data_config", "input_len"), "lag_count": get(rc, "data_config", "lag_count"),
               "train_target_start": get(rc, "data_config", "train_target_start"),
               "start": get(rc, "data_config", "start"), "train_end": get(rc, "data_config", "train_end"),
               "val_start": get(rc, "data_config", "val_start"), "val_end": get(rc, "data_config", "val_end"),
               "test_start": get(rc, "data_config", "test_start"), "test_end": get(rc, "data_config", "test_end"),
               "normalization": get(rc, "data_config", "normalization_method"),
               "optimizer": get(rc, "training_config", "optimizer"), "lr": get(rc, "training_config", "lr"),
               "weight_decay": get(rc, "training_config", "weight_decay"),
               "params": get(d, "model_measurements", "total_parameters",
                             default=get(d, "architecture", "num_params")),
               "model_config": json.dumps(rc.get("model_config", {}), sort_keys=True)}
        row.update({"re_" + k: v for k, v in recompute(df).items()})
        rows.append(row)
        preds[mj.parent.name] = df
    t = pd.DataFrame(rows)
    for k in ("mae", "mse", "rmse", "smape"):
        t["diff_" + k] = (t["re_" + k] - t["rep_" + k]).abs()
    return t, preds


def seed_means(t):
    g = t.groupby("arm")
    return g.agg(model=("model", "first"), variant=("variant", "first"), n_seeds=("seed", "nunique"),
                 seeds=("seed", lambda s: ",".join(map(str, sorted(s)))),
                 mae=("re_mae", "mean"), mse=("re_mse", "mean"), rmse=("re_rmse", "mean"),
                 smape=("re_smape", "mean"), mase=("rep_mase", "mean"))


# ---------------------------------------------------------------- run sets
sets = {}
for sub in ("aggregate-numeric-october", "aggregate-mlphead-october", "full", "sweep"):
    t, preds = load(sub)
    sets[sub] = (t, preds)
    t.drop(columns=["model_config"]).to_csv(OUT / f"runs_{sub}.csv", index=False)
    RESULT[f"recompute_{sub}"] = {
        "runs": len(t), "max_abs_diff": {k: float(t["diff_" + k].max()) for k in ("mae", "mse", "rmse", "smape")},
        "arms": int(t.arm.nunique()), "seeds_per_arm": dict(Counter(t.groupby("arm").seed.nunique())),
    }

RESULT["substituted_prediction_files"] = SUBSTITUTED
# Validate the min/max reconstruction on runs whose files already carry normalized values.
_ok = [r for r in sets["full"][0].run if f"full/{r}" not in {s["run"] for s in SUBSTITUTED}][:3]
RESULT["minmax_reconstruction_check"] = {
    r: float(np.abs(normalize_real(sets["full"][1][r])[["true_value", "true_weight", "pred_value", "pred_weight"]].to_numpy()
                    - sets["full"][1][r][["true_value", "true_weight", "pred_value", "pred_weight"]].to_numpy()).max())
    for r in _ok}

# ---------------------------------------------------------------- Table 1 / Fig 1
num = seed_means(sets["aggregate-numeric-october"][0])
mlp = seed_means(sets["aggregate-mlphead-october"][0])
num.to_csv(OUT / "table1_numeric_means.csv")
mlp.to_csv(OUT / "mlphead_means.csv")
RESULT["table1"] = num[["n_seeds", "mae", "mse", "rmse", "smape"]].to_dict(orient="index")
head_pairs = {"LSTM": ("lstm|numeric", "lstm_mlphead|numeric"), "GRU": ("gru|numeric", "gru_mlphead|numeric"),
              "Transformer": ("transformer_linhead|numeric", "transformer_mlphead|numeric"),
              "Mamba-2": ("mamba|numeric", "mamba_mlphead|numeric"), "GPT": ("gpt_linhead|numeric", "gpt_mlphead|numeric"),
              "S4 (bidir.)": ("s4_bidir|numeric", "s4_bidir_mlphead|numeric"), "S4": ("s4|numeric", "s4_mlphead|numeric")}
fig1 = {}
for lab, (lin, mh) in head_pairs.items():
    fig1[lab] = {m: (mlp.loc[mh, m] / num.loc[lin, m] - 1) * 100 for m in ("mse", "mae")}
    fig1[lab]["lin_seeds"] = int(num.loc[lin, "n_seeds"])
    fig1[lab]["mlp_seeds"] = int(mlp.loc[mh, "n_seeds"])
RESULT["fig1_pct_change_mlp_vs_linear"] = fig1
# head shape of the MLP runs (for the 128->64->2 claim)
mt = sets["aggregate-mlphead-october"][0]
RESULT["mlphead_model_configs"] = sorted(set(mt.model_config))[:6]
RESULT["linear_model_configs"] = sorted(set(sets["aggregate-numeric-october"][0].model_config))[:6]

# per-seed Wilcoxon for LSTM head change
try:
    from scipy.stats import wilcoxon
    nt = sets["aggregate-numeric-october"][0]
    a = nt[nt.arm == "lstm|numeric"].set_index("seed")
    b = mt[mt.arm == "lstm_mlphead|numeric"].set_index("seed")
    common = sorted(set(a.index) & set(b.index))
    RESULT["lstm_head_wilcoxon"] = {
        m: {"p": float(wilcoxon(b.loc[common, "re_" + m], a.loc[common, "re_" + m], method="exact").pvalue),
            "mlp_worse_seeds": int((b.loc[common, "re_" + m] > a.loc[common, "re_" + m]).sum()), "n": len(common)}
        for m in ("mse", "mae")}
except Exception as e:  # noqa: BLE001
    RESULT["lstm_head_wilcoxon"] = str(e)

# ---------------------------------------------------------------- Fig 2 sweep
st = sets["sweep"][0]
sw = st.groupby(["model", "variant", "input_len", "lag_count"]).agg(
    n=("seed", "nunique"), seeds=("seed", lambda s: ",".join(map(str, sorted(s)))), mse=("re_mse", "mean")).reset_index()
sw["mse_x100"] = sw.mse * 100
sw.to_csv(OUT / "fig2_sweep_cells.csv", index=False)
RESULT["fig2_cells"] = sw.to_dict(orient="records")
RESULT["sweep_train_target_start"] = dict(Counter(map(str, st.train_target_start)))

# ---------------------------------------------------------------- Fig 3 / Table 2
ft, fpreds = sets["full"]
full = seed_means(ft)
ma = full.loc["moving_average|none"]
full["imp_mse"] = (ma.mse - full.mse) / ma.mse * 100
full["imp_mae"] = (ma.mae - full.mae) / ma.mae * 100
full["rank_mse"] = full.mse.rank(method="min").astype(int)
full["rank_mae"] = full.mae.rank(method="min").astype(int)

# point losses averaged over seeds, indexed by KEY
def point_losses(metric):
    out = {}
    for arm, g in ft.groupby("arm"):
        acc = None
        for run in g.run:
            df = fpreds[run].set_index(KEY).sort_index()
            e = np.stack([df.pred_value - df.true_value, df.pred_weight - df.true_weight], axis=1)
            loss = pd.Series((e ** 2 if metric == "mse" else np.abs(e)).mean(axis=1), index=df.index)
            acc = loss if acc is None else acc + loss
        out[arm] = acc / g.run.nunique()
    return out


def wild_cluster_p(d, states):
    labels, inv = np.unique(states, return_inverse=True)
    G, n = len(labels), len(d)

    def tstat(x):
        m = x.mean()
        s = np.bincount(inv, weights=x - m, minlength=G)
        return m / np.sqrt(G / (G - 1) * (s ** 2).sum() / n ** 2)

    t0 = tstat(d)
    hits = sum(abs(tstat(d * np.asarray(w)[inv])) >= abs(t0) - 1e-12
               for w in itertools.product((1.0, -1.0), repeat=G))
    return hits / 2 ** G, t0, G


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        q[i] = prev
    return q


baselines = {"moving_average|none", "random_walk|none", "seasonal_naive|none"}
for metric in ("mse", "mae"):
    pl = point_losses(metric)
    base = pl["moving_average|none"]
    arms = [a for a in full.index if a not in baselines]
    ps = []
    for a in arms:
        assert pl[a].index.equals(base.index), a
        d = (base - pl[a]).to_numpy()
        p, t0, G = wild_cluster_p(d, base.index.get_level_values("State").to_numpy())
        ps.append(p)
        full.loc[a, f"p_{metric}"] = p
        full.loc[a, f"beats_{metric}"] = d.mean() > 0
        full.loc[a, "n_clusters"] = G
    q = bh(ps)
    for a, qq in zip(arms, q):
        full.loc[a, f"q_{metric}"] = qq
    full[f"dagger_{metric}"] = full[f"beats_{metric}"].fillna(False).astype(bool) & (full[f"p_{metric}"] < 0.05)
full.to_csv(OUT / "fig3_table2_full_arms.csv")
RESULT["full_arms"] = full.reset_index().to_dict(orient="records")

# ---------------------------------------------------------------- MASE from raw data
raw = RAW
anyrun = fpreds[ft.run.iloc[0]]
lanes = anyrun[["State", "Commodity", "Import/Export"]].drop_duplicates()
tr = raw[(raw.Time >= "2009-01-01") & (raw.Time <= "2023-12-01")].sort_values("Time")
scale = {}
for (s, c, f), g in tr.groupby(["State", "Commodity", "Import/Export"]):
    v, w = g.Value.to_numpy(float), g.Weight.to_numpy(float)
    scale[(s, c, f)] = (np.abs(v[12:] - v[:-12]).mean(), np.abs(w[12:] - w[:-12]).mean())
# truth identity: predictions' real truths vs raw dataset
test = raw[(raw.Time >= "2025-01-01") & (raw.Time <= "2025-05-01")].copy()
test["Time"] = test.Time.dt.strftime("%Y-%m-%d")
chk = anyrun.merge(test, on=KEY, how="left")
RESULT["truth_identity"] = {
    "rows": len(chk), "unmatched": int(chk.Value.isna().sum()),
    "max_abs_value_diff": float((chk.true_value_real - chk.Value).abs().max()),
    "max_abs_weight_diff": float((chk.true_weight_real - chk.Weight).abs().max())}


def mase_real(df):
    vals_v, vals_w = [], []
    for (s, c, f), g in df.groupby(["State", "Commodity", "Import/Export"]):
        sv, sw_ = scale[(s, c, f)]
        vals_v.append(np.abs(g.pred_value_real - g.true_value_real).mean() / sv)
        vals_w.append(np.abs(g.pred_weight_real - g.true_weight_real).mean() / sw_)
    return (np.mean(vals_v) + np.mean(vals_w)) / 2


ft["re_mase"] = [mase_real(fpreds[r]) for r in ft.run]
full["mase_recomputed"] = ft.groupby("arm").re_mase.mean()
RESULT["mase_max_abs_diff_vs_reported"] = float((ft.re_mase - ft.rep_mase).abs().max())
full.to_csv(OUT / "fig3_table2_full_arms.csv")

# ---------------------------------------------------------------- cohort / dataset claims
g = raw[(raw.Time >= "2009-01-01") & (raw.Time <= "2025-05-01")]
lane_cols = ["State", "Commodity", "Import/Export"]
months = g.Time.nunique()
any_nonzero = g.groupby(lane_cols).Value.apply(lambda s: (s != 0).any())
any_rows = g.groupby(lane_cols).SourceRows.sum() > 0
trn = g[g.Time <= "2023-12-01"]
zero_frac = trn.groupby(lane_cols).Value.apply(lambda s: (s == 0).mean())
kept = set(map(tuple, lanes.to_numpy()))
retained = set(zero_frac[zero_frac <= 0.05].index)
RESULT["cohort"] = {
    "months_2009_01_to_2025_05": int(months), "states": int(g.State.nunique()),
    "commodities_in_source": int(g.Commodity.nunique()),
    "lanes_in_grid": int(len(any_nonzero)), "lanes_any_nonzero_value": int(any_nonzero.sum()),
    "lanes_any_source_rows": int(any_rows.sum()),
    "train_months": int(trn.Time.nunique()),
    "retained_zero_frac_le_5pct": len(retained), "retained_equals_prediction_lanes": retained == kept,
    "prediction_lanes": len(kept),
    "retained_commodities": len({c for _, c, _ in kept}), "retained_states": len({s for s, _, _ in kept}),
    "density_kept_over_states_x_commodities_x_2": len(kept) / (len({s for s, _, _ in kept}) * len({c for _, c, _ in kept}) * 2),
}

# ---------------------------------------------------------------- hyperparameters
hp = defaultdict(Counter)
for sub in ("full", "sweep", "aggregate-numeric-october", "aggregate-mlphead-october"):
    t = sets[sub][0]
    for _, r in t.iterrows():
        mc = json.loads(r.model_config)
        for k in ("num_layers", "n_layers", "hidden_size", "d_model", "dropout", "embedding_dims"):
            if k in mc:
                hp[f"{sub}:{k}"][json.dumps(mc[k], sort_keys=True)] += 1
        for k in ("optimizer", "lr", "weight_decay", "start", "train_end", "val_start", "val_end",
                  "test_start", "test_end", "normalization", "train_target_start"):
            hp[f"{sub}:{k}"][str(r[k])] += 1
    hp[f"{sub}:seeds"] = Counter(map(str, sorted(t.seed.unique())))
    hp[f"{sub}:params_range"] = Counter([f"{int(t.params.min()) if t.params.notna().any() else None}-{int(t.params.max()) if t.params.notna().any() else None}"])
RESULT["hyperparams"] = {k: dict(v) for k, v in hp.items()}

# ---------------------------------------------------------------- SHAP (Fig 4)
shap = {}
pred_lanes = anyrun.copy()
for name in ("lstm", "s4"):
    z = np.load(SNAP / "shap" / f"{name}_cross_attention_2d_seed26.npz", allow_pickle=False)
    own, src, feats, keys, tg = z["own"].astype(np.float64), z["src"].astype(np.float64), list(z["feat_cols"]), list(z["combo_keys"]), z["targets"]
    info = {"own_shape": list(own.shape), "src_shape": list(src.shape), "feat_cols": feats, "status": str(z["status"])}
    own_net = own.sum(axis=3)                         # (L, M, O, F) net over the input window
    tgt_keys = [keys[i].split("|") for i in tg]
    rel = np.zeros((len(tg), len(keys)), dtype=object)
    for li, (s, c, f) in enumerate(tgt_keys):
        for j, k in enumerate(keys):
            s2, c2, f2 = k.split("|")
            rel[li, j] = ("self" if j == tg[li]
                          else "other states, same commodity" if (s2 != s and c2 == c and f2 == f)
                          else "same state & commodity, other flow" if (s2 == s and c2 == c)
                          else "same state, other commodities" if s2 == s else "unrelated")
    # Shares: |net over the input window| per own feature, |net over a lane group and its
    # channels| per relation group, each per (lane, month, output); share = sum / grand total.
    own_mag = np.abs(own_net)
    gmag = {g: np.abs(np.stack([src[li][:, :, rel[li] == g, :].sum(axis=(2, 3)) for li in range(len(tg))]))
            for g in sorted(set(rel.ravel()) - {"self"})}
    total = own_mag.sum() + sum(v.sum() for v in gmag.values())
    shares = {f: own_mag[..., i].sum() / total for i, f in enumerate(feats)}
    shares.update({g: v.sum() / total for g, v in gmag.items()})
    info["shares_top8"] = dict(sorted(shares.items(), key=lambda kv: -kv[1])[:8])
    # dollar magnitude: |net attribution| on the VALUE output x lane value scale (train min-max range)
    lane_scale = []
    for s, c, f in tgt_keys:
        h = tr[(tr.State == s) & (tr.Commodity == c) & (tr["Import/Export"] == f)].Value
        lane_scale.append(h.max() - h.min())
    lane_scale = np.asarray(lane_scale)[:, None]
    ci = feats.index("cos_month") if "cos_month" in feats else None
    dollars = {}
    if ci is not None:
        dollars["cos_month"] = float((np.abs(own_net[:, :, 0, ci]) * lane_scale).mean())
    grp = (rel == "other states, same commodity")
    src_v = src[:, :, 0, :, :].sum(axis=3)            # (L, M, J) net over channels
    net_grp = np.stack([src_v[li][:, grp[li]].sum(axis=1) for li in range(len(tg))])   # (L, M)
    dollars["other_states_same_commodity_net"] = float((np.abs(net_grp) * lane_scale).mean())
    dollars["other_states_same_commodity_sumabs"] = float((np.stack(
        [np.abs(src[li, :, 0][:, grp[li], :]).sum(axis=(1, 2)) for li in range(len(tg))]) * lane_scale).mean())
    info["dollars"] = dollars
    info["lanes"], info["months"] = len(tg), own.shape[1]
    shap[name] = info
RESULT["shap"] = shap

(OUT / "computed.json").write_text(json.dumps(RESULT, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("wrote", OUT / "computed.json")
for k, v in RESULT.items():
    if k.startswith("recompute_"):
        print(k, v)
print("truth_identity", RESULT["truth_identity"])
print("cohort", RESULT["cohort"])
print("mase diff", RESULT["mase_max_abs_diff_vs_reported"])
