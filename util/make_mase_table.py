#!/usr/bin/env python
"""Source-validated scale-free metrics beside exploratory state-level ranks.
MASE denominators are training seasonal errors. Out-of-sample seasonal naive
MASE need not equal one. Seed intervals describe initialization variability.
"""
from collections import defaultdict
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.exp2_artifacts import validated_exp2_artifacts
from util.make_significance import load_model_losses, significance_table
RUNS = ROOT / "outputs/full"
OUTDIR = ROOT / "graphs/_analysis"


def seed_mean_metrics():
    acc = defaultdict(lambda: defaultdict(list))
    for run, path, payload in validated_exp2_artifacts(ROOT, RUNS):
        arm = f"{run.model}_{run.variant}"
        for key in ("mase", "rmsse", "smape"):
            acc[arm][key].append(float(payload.get("metrics", payload)[key]))
    return {arm: {key: sum(values) / len(values) for key, values in metrics.items()}
            for arm, metrics in acc.items()}


def main():
    table = significance_table(load_model_losses())
    metrics = seed_mean_metrics()
    for key in ("mase", "rmsse"):
        table[key] = table.run.map(lambda arm: metrics[arm][key])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTDIR / "mase_table.csv", index=False)
    columns = ["run", "mean_norm_mse", "ci95_halfwidth", "mase", "rmsse",
               "mean_rank", "within_cd_of_reference", "n_blocks", "inference_status"]
    # Avoid an optional tabulate dependency for a plain Markdown artifact.
    lines = ["Exploratory state-block analysis; within-CD does not mean equivalent.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(row[key]) for key in columns) + " |"
              for _, row in table.iterrows()]
    (OUTDIR / "mase_table.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
