#!/usr/bin/env python3
"""Illustrate the trade dataset: aggregate + top-3-commodity time series
for exports and imports. Values/Time parsed exactly as src/data/dataloader.py.

Outputs 4 PNGs (aggregate + top-3, for exports and imports) plus a combined
2x2 overview, into the directory given by --outdir (default: dataset_plots/).
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- value columns, matching src/data/dataloader.py ------------------------
EXPORT_VALUE = "Containerized Vessel Total Exports Value ($US)"
IMPORT_VALUE_PRI = "Vessel Value ($US)"
IMPORT_VALUE_ALT = "Customs Containerized Vessel Value (Gen) ($US)"

MONTH_MAP = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# --- dataviz categorical palette (validated slots) -------------------------
BLUE, GREEN, RED = "#2a78d6", "#008300", "#e34948"
INK, MUTED, GRID = "#1a1a19", "#6b6b66", "#e7e7e2"


def parse_month(s):
    """Replicate dataloader.parse_month: 'YYYY-MM' or 'YY-Mon' (e.g. 8-Jan)."""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    try:
        dt = pd.to_datetime(s, format="%Y-%m", errors="raise")
        return pd.Timestamp(dt.year, dt.month, 1)
    except Exception:
        pass
    try:
        yy, mon = s.split("-")
        return pd.Timestamp(int(yy) + 2000, MONTH_MAP[mon.capitalize()], 1)
    except Exception:
        pass
    try:
        dt = pd.to_datetime(s)
        return pd.Timestamp(dt.year, dt.month, 1)
    except Exception:
        return pd.NaT


def numeric_clean(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ""),
                         errors="coerce").fillna(0.0)


def load_flow(folder, value_pri, value_alt=None):
    rows = []
    for fp in sorted(Path(folder).glob("*.csv")):
        df = pd.read_csv(fp, on_bad_lines="skip", dtype=str)
        vcol = value_pri if value_pri in df.columns else value_alt
        out = pd.DataFrame({
            "Time": df["Time"].apply(parse_month),
            "Commodity": df["Commodity"].astype(str).str.strip(),
            "Value": numeric_clean(df[vcol]),
        })
        rows.append(out)
    df = pd.concat(rows, ignore_index=True)
    return df.dropna(subset=["Time"])


def usd_fmt(x, _):
    ax = abs(x)
    if ax >= 1e9:
        return f"${x/1e9:.1f}B"
    if ax >= 1e6:
        return f"${x/1e6:.0f}M"
    if ax >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def short_commodity(name, n=34):
    name = " ".join(name.split())
    return name if len(name) <= n else name[: n - 1] + "…"


def style_ax(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))


def monthly_total(df):
    return df.groupby("Time")["Value"].sum().sort_index()


def top_commodities(df, k=3):
    tot = df.groupby("Commodity")["Value"].sum().sort_values(ascending=False)
    return tot.head(k).index.tolist()


def plot_aggregate(ax, series, title, color):
    ax.plot(series.index, series.values, color=color, linewidth=2, zorder=3)
    ax.fill_between(series.index, series.values, color=color, alpha=0.10, zorder=2)
    style_ax(ax)
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left", pad=10)


def plot_top3(ax, df, commodities, title):
    colors = [BLUE, GREEN, RED]
    for comm, col in zip(commodities, colors):
        s = monthly_total(df[df["Commodity"] == comm])
        s = s.reindex(sorted(df["Time"].unique())).fillna(0.0).sort_index()
        ax.plot(s.index, s.values, color=col, linewidth=1.8, zorder=3,
                label=short_commodity(comm))
        # direct end-label (contrast-floor relief for low-contrast hues)
        ax.annotate(short_commodity(comm, 22),
                    xy=(s.index[-1], s.values[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=col, fontweight="bold")
    style_ax(ax)
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    for t in leg.get_texts():
        t.set_color(INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default="data/exports")
    ap.add_argument("--imports", default="data/imports")
    ap.add_argument("--outdir", default="graphs/dataset")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading exports…")
    exp = load_flow(args.exports, EXPORT_VALUE)
    print("Loading imports…")
    imp = load_flow(args.imports, IMPORT_VALUE_PRI, IMPORT_VALUE_ALT)

    exp_total, imp_total = monthly_total(exp), monthly_total(imp)
    exp_top, imp_top = top_commodities(exp), top_commodities(imp)
    span = f"{exp_total.index.min():%b %Y} – {exp_total.index.max():%b %Y}"
    print(f"Span: {span}")
    print(f"Total export value: ${exp['Value'].sum()/1e9:.1f}B  |  "
          f"import value: ${imp['Value'].sum()/1e9:.1f}B")
    print("Top-3 exports:", exp_top)
    print("Top-3 imports:", imp_top)

    plt.rcParams["font.family"] = "DejaVu Sans"

    specs = [
        ("exports_aggregate.png", "agg", exp_total,
         f"Total containerized vessel exports — all 8 states, all commodities\n{span}", BLUE),
        ("exports_top3_commodities.png", "top", (exp, exp_top),
         "Top 3 export commodities by total value (monthly)"),
        ("imports_aggregate.png", "agg", imp_total,
         f"Total vessel imports — all 8 states, all commodities\n{span}", GREEN),
        ("imports_top3_commodities.png", "top", (imp, imp_top),
         "Top 3 import commodities by total value (monthly)"),
    ]
    for spec in specs:
        fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
        if spec[1] == "agg":
            _, _, series, title, color = spec
            plot_aggregate(ax, series, title, color)
        else:
            _, _, (df, top), title = spec
            plot_top3(ax, df, top, title)
        ax.set_xlabel("")
        fig.text(0.005, 0.01, "US agricultural trade data · data/README.md",
                 fontsize=7, color=MUTED)
        fig.tight_layout()
        fig.savefig(out / spec[0], bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", out / spec[0])

    # combined 2x2 overview
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=150)
    plot_aggregate(axes[0, 0], exp_total, "Total exports (all states & commodities)", BLUE)
    plot_top3(axes[0, 1], exp, exp_top, "Top 3 export commodities")
    plot_aggregate(axes[1, 0], imp_total, "Total imports (all states & commodities)", GREEN)
    plot_top3(axes[1, 1], imp, imp_top, "Top 3 import commodities")
    fig.suptitle(f"US agricultural trade dataset — exports & imports  ({span})",
                 fontsize=16, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "overview_2x2.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out / "overview_2x2.png")


if __name__ == "__main__":
    main()
