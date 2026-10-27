#!/usr/bin/env python
"""Generate LaTeX from freshly validated Exp 1/2 artifacts.
Bold metric cells are descriptive minima. Bold names mark the exploratory
state-rank critical-difference interval, not equivalence. Daggers denote a
positive lane-weighted difference passing Holm correction against moving average.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
sys.path.insert(0, str(ROOT))

MODEL_DISPLAY = {
    "lstm": "LSTM", "gru": "GRU", "transformer": "Transformer", "gpt": "GPT",
    "mamba": "Mamba-2", "s4": "S4", "s4_bidir": "S4-Bidirectional", "mamba_nd": "Mamba-ND", "s4nd": "S4ND",
    "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "moving_average": "Moving Average", "random_walk": "Random Walk",
    "seasonal_naive": "Seasonal Naive",
}
VARIANT_DISPLAY = {
    "onehot": "one-hot", "embeddings": "embeddings", "none": "--",
    "cross_attention_2d": "axial CA 2-D", "cross_attention_3d": "axial CA 3-D",
    "cross_attention_4d": "axial CA 4-D",
    "2d": "native 2-D", "2d_state": "native 2-D (state)",
    "3d": "native 3-D", "4d": "native 4-D",
    "cross_attention": "flat CA", "film_attention": "FiLM",
}
FAMILY = {
    "lstm": "RNN", "gru": "RNN", "transformer": "Attention", "gpt": "Attention",
    "mamba": "SSM", "s4": "SSM", "s4_bidir": "SSM", "mamba_nd": "SSM", "s4nd": "SSM",
    "xgboost": "Tree", "lightgbm": "Tree", "random_forest": "Tree",
    "moving_average": "Baseline", "random_walk": "Baseline",
    "seasonal_naive": "Baseline",
}
FAM_ORDER = ["SSM", "Attention", "RNN", "Tree", "Baseline"]
STRUCT = {  # both naming schemes onto one structural axis
    "onehot": "F1", "embeddings": "F2",
    "cross_attention_2d": "D2", "2d": "D2", "2d_state": "D2S",
    "cross_attention_3d": "D3", "3d": "D3",
    "cross_attention_4d": "D4", "4d": "D4",
}
STRUCT_COLS = [("F1", "Flat (one-hot)"), ("F2", "Flat (embed.)"),
               ("D2", "2-D"), ("D2S", "2-D state"), ("D3", "3-D"), ("D4", "4-D")]
HEADER = ("% AUTO-GENERATED from validated artifacts by util/make_paper_tables.py\n"
          "% -- do not hand-edit; re-run the script after run changes.\n")


def fmt(x, nd=6):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def bold_min(series):
    """Set of index labels holding the column minimum (all ties)."""
    m = series.min()
    return set(series[np.isclose(series, m)].index)


def load():
    from scripts.build_paper_results import _exp1_rows, _exp2_rows, _aggregate_seeds
    from util.make_significance import load_model_losses, significance_table
    from util.make_vs_baseline import baseline_table
    exp1_rows, _ = _exp1_rows(ROOT)
    pc = _aggregate_seeds(_exp2_rows(ROOT))
    pc["arm"] = pc.model.astype(str) + "_" + pc.variant.astype(str)
    data = load_model_losses()
    sig = significance_table(data)
    tied = dict(zip(sig.run, sig.within_cd_of_reference))
    rank = dict(zip(sig.run, sig.mean_rank))
    vb = baseline_table(data)
    dagger = {r.arm: bool(r.beats_baseline and r.p_holm < .05) for _, r in vb.iterrows()}
    impr = dict(zip(vb.arm, vb.improvement_pct))
    # exp1_table aggregates raw seeds itself; do not feed it a seedless summary.
    return pc, tied, rank, dagger, impr, pd.DataFrame(exp1_rows)


def exp1_table(e1):
    g = (e1.groupby(["model", "variant"])
           .agg(mae=("mae", "mean"), mse=("norm_mse_combined", "mean"),
                rmse=("rmse", "mean"), smape=("smape", "mean"),
                n_seeds=("seed", "nunique"))
           .reset_index().sort_values("mse").reset_index(drop=True))
    bolds = {c: bold_min(g[c]) for c in ("mae", "mse", "rmse", "smape")}
    lines = [HEADER,
             r"\begin{table}[h]",
             r"\caption{Experiment 1 (aggregated series). Stochastic arms"
             r" and tree arms are means over the 25 declared seeds;"
             r" deterministic naive baselines are evaluated once."
             r"}% TODO(author): finalize caption",
             r"\label{tab:exp1agg}", r"\centering", r"\footnotesize",
             r"\begin{tabular*}{\hsize}{@{\extracolsep{\fill}}llrrrr@{}}",
             r"\toprule",
             r"Model & Variant & MAE & MSE & RMSE & sMAPE\% \\",
             r"\midrule"]
    for i, r in g.iterrows():
        cells = []
        for c, nd in (("mae", 6), ("mse", 6), ("rmse", 6), ("smape", 2)):
            v = fmt(r[c], nd)
            cells.append(rf"\textbf{{{v}}}" if i in bolds[c] else v)
        lines.append(f"{MODEL_DISPLAY.get(r.model, r.model)} & "
                     f"{VARIANT_DISPLAY.get(r.variant, r.variant)} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def leaderboard_table(pc, tied, rank, dagger):
    t = pc.sort_values("norm_mse_combined").reset_index(drop=True)
    bolds = {c: bold_min(t[c]) for c in
             ("mae", "norm_mse_combined", "rmse", "mase", "smape")}
    lines = [HEADER,
             r"\begin{table}[p]",
             r"\caption{Experiment 2, all arms. Bold arm name = exploratorily"
             r" within the state-rank CD of the reference (not equivalence); $\dagger$ = beats the moving average"
             r" at Holm-adjusted $p<0.05$."
             r"}% TODO(author): finalize caption",
             r"\label{tab:fullleaderboard}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular*}{\hsize}"
             r"{@{\extracolsep{\fill}}rllrrrrrr@{}}",
             r"\toprule",
             r"\# & Model & Encoder & MAE & MSE & RMSE & MASE & sMAPE\% &"
             r" Rank \\",
             r"\midrule"]
    for i, r in t.iterrows():
        name = MODEL_DISPLAY.get(r.model, r.model)
        if tied.get(r.arm, False):
            name = rf"\textbf{{{name}}}"
        cells = []
        for c, nd in (("mae", 6), ("norm_mse_combined", 6), ("rmse", 6),
                      ("mase", 3), ("smape", 2)):
            v = fmt(r[c], nd)
            if i in bolds[c]:
                v = rf"\textbf{{{v}}}"
            if c == "norm_mse_combined" and dagger.get(r.arm, False):
                v += r"$^{\dagger}$"
            cells.append(v)
        rk = rank.get(r.arm, np.nan)
        lines.append(f"{i + 1} & {name} & "
                     f"{VARIANT_DISPLAY.get(r.variant, r.variant)} & "
                     + " & ".join(cells)
                     + f" & {fmt(rk, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def matrix_table(pc, tied, dagger):
    t = pc.copy()
    t["struct"] = t.variant.map(STRUCT)
    t = t.dropna(subset=["struct"])
    models = [m for m in ("s4", "s4_bidir", "mamba_nd", "mamba", "s4nd", "transformer",
                          "gpt", "lstm", "gru", "xgboost", "lightgbm",
                          "random_forest") if m in set(t.model)]
    lines = [HEADER,
             r"\begin{table}[t]",
             r"\caption{Normalized MSE by backbone and structural dimension"
             r" of the combination encoder. Bold = within the state-rank CD"
             r" of the reference (exploratory, not equivalence); $\dagger$ = beats the moving average at"
             r" Holm-adjusted $p<0.05$; -- = encoder not supported."
             r"}% TODO(author): finalize caption; note MA reference"
             r" MSE in the text",
             r"\label{tab:structmatrix}", r"\centering", r"\footnotesize",
             r"\begin{tabular*}{\hsize}{@{\extracolsep{\fill}}lrrrrrr@{}}",
             r"\toprule",
             "Model & " + " & ".join(lbl for _, lbl in STRUCT_COLS) + r" \\",
             r"\midrule"]
    prev_fam = None
    for m in models:
        fam = FAMILY[m]
        if prev_fam and fam != prev_fam:
            lines.append(r"\midrule")
        prev_fam = fam
        row = [MODEL_DISPLAY.get(m, m)]
        sub = t[t.model == m]
        for key, _ in STRUCT_COLS:
            cand = sub[sub.struct == key]
            if cand.empty:
                row.append("--")
                continue
            if len(cand) != 1:
                raise RuntimeError("Multiple arms map to one table cell")
            r = cand.iloc[0]
            v = fmt(r.norm_mse_combined, 6)
            if tied.get(r.arm, False):
                v = rf"\textbf{{{v}}}"
            if dagger.get(r.arm, False):
                v += r"$^{\dagger}$"
            row.append(v)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def synthesis_table(pc, tied, rank, dagger, impr):
    t = pc.copy()
    t["family"] = t.model.map(FAMILY)
    t["r_mse"] = t.norm_mse_combined.rank(method="min").astype(int)
    t["r_mae"] = t.mae.rank(method="min").astype(int)
    lines = [HEADER,
             r"\begin{table}[t]",
             r"\caption{Synthesis: best arm per model family. $\Delta$MA ="
             r" improvement in mean normalized MSE over the moving average;"
             r" $\dagger$ = positive lane-weighted difference at Holm-adjusted $p<0.05$."
             r" Within-CD = exploratory state-rank interval, not equivalence.}% TODO(author): finalize caption AND fill the"
             r" `Recommended when' column -- interpretation is yours",
             r"\label{tab:synthesis}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular*}{\hsize}"
             r"{@{\extracolsep{\fill}}llrrrrcl@{}}",
             r"\toprule",
             r"Family & Best arm & MSE rank & MAE rank & MASE &"
             r" $\Delta$MA\,\% & Within-CD & Recommended when \\",
             r"\midrule"]
    for fam in FAM_ORDER:
        sub = t[t.family == fam]
        if sub.empty:
            continue
        if fam == "Baseline":
            sub = sub[sub.model == "moving_average"]
        r = sub.loc[sub.norm_mse_combined.idxmin()]
        arm_disp = (f"{MODEL_DISPLAY.get(r.model, r.model)} "
                    f"{VARIANT_DISPLAY.get(r.variant, r.variant)}").replace(
                    " --", "")
        d = impr.get(r.arm)
        dtxt = ("ref." if fam == "Baseline" else
                ("--" if d is None else f"{d:+.2f}"
                 + (r"$^{\dagger}$" if dagger.get(r.arm, False) else "")))
        lines.append(
            f"{fam} & {arm_disp} & {int(r.r_mse)} & {int(r.r_mae)} & "
            f"{fmt(r.mase, 3)} & {dtxt} & "
            + (r"\checkmark" if tied.get(r.arm, False) else "--")
            + " &  \\\\  % TODO(author): recommendation")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def selected_arms_table(pc, tied, rank, dagger):
    """Condensed alternative to the full leaderboard: tied group + best arm
    per family + best-MASE arm + baselines (~13 rows). The '#' column keeps
    each arm's position in the FULL ordering, so the 12 -> 38 gap stays
    visible. Pair with a data-availability sentence for the full table."""
    t = pc.sort_values("norm_mse_combined").reset_index(drop=True)
    t["family"] = t.model.map(FAMILY)
    keep = set(a for a in t.arm if tied.get(a, False))
    keep |= set(t.loc[t.groupby("family").norm_mse_combined.idxmin().values].arm)
    keep.add(t.loc[t.mase.idxmin()].arm)
    keep |= set(t[t.family == "Baseline"].arm)
    sel = t[t.arm.isin(keep)].reset_index(drop=True)
    pos = {a: i + 1 for i, a in enumerate(t.arm)}
    bolds = {c: bold_min(sel[c]) for c in
             ("mae", "norm_mse_combined", "rmse", "mase")}
    lines = [HEADER,
             r"\begin{table}[t]",
             r"\caption{Selected arms: the exploratory within-CD group, the"
             r" best arm per family, the best-MASE arm, and the naive"
             r" baselines. Bold arm name = within CD (not equivalence); $\dagger$ ="
             r" beats the moving average at Holm-adjusted $p<0.05$. Full"
             r" per-arm results are available in the project repository."
             r"}% TODO(author): finalize caption + repository reference",
             r"\label{tab:selectedarms}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular*}{\hsize}{@{\extracolsep{\fill}}rllrrrrrr@{}}",
             r"\toprule",
             r"\# & Model & Encoder & MAE & MSE & RMSE & MASE & Rank \\",
             r"\midrule"]
    for i, r in sel.iterrows():
        name = MODEL_DISPLAY.get(r.model, r.model)
        if tied.get(r.arm, False):
            name = rf"\textbf{{{name}}}"
        cells = []
        for c, nd in (("mae", 6), ("norm_mse_combined", 6), ("rmse", 6),
                      ("mase", 3)):
            v = fmt(r[c], nd)
            if i in bolds[c]:
                v = rf"\textbf{{{v}}}"
            if c == "norm_mse_combined" and dagger.get(r.arm, False):
                v += r"$^{\dagger}$"
            cells.append(v)
        lines.append(f"{pos[r.arm]} & {name} & "
                     f"{VARIANT_DISPLAY.get(r.variant, r.variant)} & "
                     + " & ".join(cells) + f" & {fmt(rank.get(r.arm), 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def encoder_ratio_table(pc):
    piv = pc[pc.variant.isin(["onehot", "embeddings"])].pivot_table(
        index="model", columns="variant",
        values=["mae", "norm_mse_combined"])
    piv = piv.dropna()
    rows = []
    for m, r in piv.iterrows():
        rows.append((m, r[("mae", "onehot")] / r[("mae", "embeddings")],
                     r[("norm_mse_combined", "onehot")]
                     / r[("norm_mse_combined", "embeddings")]))
    rows.sort(key=lambda x: list(MODEL_DISPLAY).index(x[0]))
    lines = [HEADER,
             r"\begin{table}[h]",
             r"\caption{Encoder error ratio (one-hot / embeddings), all"
             r" backbones supporting both encoders."
             r"}% TODO(author): finalize caption",
             r"\label{tab:encoderratio}", r"\centering", r"\footnotesize",
             r"\begin{tabular*}{\hsize}{@{\extracolsep{\fill}}lrr@{}}",
             r"\toprule", r"Model & MAE ratio & MSE ratio \\", r"\midrule"]
    for m, ra, rm in rows:
        lines.append(f"{MODEL_DISPLAY.get(m, m)} & {ra:.3f} & {rm:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pc, tied, rank, dagger, impr, e1 = load()
    outputs = {
        "exp1_aggregated.tex": exp1_table(e1),
        "exp2_full_leaderboard.tex": leaderboard_table(pc, tied, rank, dagger),
        "exp2_matrix.tex": matrix_table(pc, tied, dagger),
        "exp2_synthesis.tex": synthesis_table(pc, tied, rank, dagger, impr),
        "exp2_selected_arms.tex": selected_arms_table(pc, tied, rank, dagger),
        "exp2_encoder_ratio.tex": encoder_ratio_table(pc),
    }
    for name, content in outputs.items():
        (OUT / name).write_text(content)
        print(f"[tables] wrote {name} ({content.count(chr(10))} lines)")
    print(f"[tables] arms in leaderboard: {len(pc)}; "
          f"within-CD group size: {sum(tied.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
