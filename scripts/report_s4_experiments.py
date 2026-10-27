#!/usr/bin/env python
"""Read-only S4/S4ND inventory, resolved from the canonical launch contracts.

No model fitting, data rewriting, or result reuse. JSON includes full contracts;
Markdown groups seed replicas but lists every window/lag configuration.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.exp2_protocol import expected_runs, expected_contract
from src.utils.exp3_protocol import coordinates, expected_contract as sweep_contract, run_name
from src.utils.experiment_manifest import (
    load_manifest, exp1_expected_runs, exp1_protocol, exp1_source_domain,
)
from src.utils.run_contract import expected_exp1_run_contract


CONTENTS = {
    ("s4", "embeddings"): "State/commodity/flow embeddings (8/32/2) + temporal S4",
    ("s4", "onehot"): "One-hot state/commodity/flow IDs + temporal S4",
    ("s4", "cross_attention_2d"): "State axial attention + commodity/flow embeddings (32/2) + temporal S4",
    ("s4", "cross_attention_3d"): "State/commodity axial attention + flow embedding (2) + temporal S4",
    ("s4", "cross_attention_4d"): "State/commodity/flow axial attention + temporal S4; no ID embeddings",
    ("s4nd", "2d_state"): "S4ND over state x time; commodity-flow groups in batch; embeddings (32/2)",
    ("s4nd", "2d"): "S4ND over state x commodity-flow-group x time; embeddings (32/2)",
    ("s4nd", "3d"): "S4ND over state x commodity x time; flow in batch + embedding (2)",
    ("s4nd", "4d"): "S4ND over state x commodity x flow x time; no ID embeddings",
}
GOALS = {
    1: "Aggregate forecasting and seed stability",
    2: "Compare categorical encoding and cross-series structure",
    3: "Measure sensitivity to history window and lag features",
}
CONTENTS.update({("s4_bidir", variant): contents
                 for (model, variant), contents in list(CONTENTS.items()) if model == "s4"})


def inventory():
    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    grouped = {}

    def add(experiment, directory, contract):
        identity, data = contract["identity"], contract["data_config"]
        model, variant = identity["model"], identity["variant"]
        key = (experiment, model, variant, data["input_len"], data["lag_count"])
        normalized = copy.deepcopy(contract)
        normalized["identity"].pop("seed")
        if key not in grouped:
            external = "model" in contract["model_config"]
            architecture = contract["model_config"]["model"] if external else contract["model_config"]
            layer = architecture["layer"] if external else architecture
            training = contract["training_config"]
            batch = (training.get("loader_train", {}).get("batch_size", training["loader"]["batch_size"])
                     if external else training["batch_size"])
            grouped[key] = {
                "experiment": experiment, "model": model, "variant": variant,
                "goal": GOALS[experiment], "contents": CONTENTS[(model, variant)],
                "window": data["input_len"], "lags": data["lag_count"],
                "layers": architecture["n_layers"], "width": architecture["d_model"],
                "state_size": layer["d_state"], "n_ssm": layer["n_ssm"],
                "dropout": architecture["dropout"], "bidirectional": layer.get("bidirectional", False),
                "kernel_axes": layer.get("dim", 1), "batch_size": batch,
                "batch_by_target_month": data["batch_by_target_month"],
                "runner": "external/s4/train.py" if external else "scripts/train.py (modular S4 adapter)",
                "representative_contract": contract, "runs": [],
                "_normalized": normalized,
            }
        elif normalized != grouped[key]["_normalized"]:
            raise ValueError(f"Unexpected seed-dependent configuration: {key}")
        grouped[key]["runs"].append({"seed": identity["seed"], "directory": directory})

    for run in exp1_expected_runs(manifest):
        if run.model in {"s4", "s4_bidir"}:
            add(1, run.directory_name, expected_exp1_run_contract(
                ROOT, run, exp1_protocol(manifest), exp1_source_domain(manifest)))
    for run in expected_runs(manifest):
        if run.model in {"s4", "s4_bidir", "s4nd"}:
            add(2, run.directory_name, expected_contract(ROOT, run, manifest))
    for model, window, lag, seed in coordinates():
        if model in {"s4", "s4_bidir"}:
            add(3, run_name(model, window, lag, seed), sweep_contract(model, window, lag, seed))
    rows = [grouped[key] for key in sorted(grouped)]
    for row in rows:
        row.pop("_normalized")
    return rows


def markdown(rows):
    lines = ["# S4 experiment goals, contents and configurations", "",
             "Generated from the current canonical launch contracts by `python scripts/report_s4_experiments.py`.",
             "Use `--format json` for complete model, data, training, selection and source settings.", "",
             "All rows use the actual modular `external/s4/src/models/sequence` implementation.",
             "Forecasting/axial adapters and existing S4ND padding corrections are local integration; see `external/s4/UPSTREAM.md`.",
             "Configuration inventory is not evidence of completed training or GPU qualification.", "",
             "| Experiment / model | Direction | Goal | Contents | Layers / width / state | Window / lags | Batch | Seeds / runs |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        direction = "bidirectional" if row["bidirectional"] else "causal"
        lines.append(f"| Exp {row['experiment']} - {row['model']} {row['variant']} | {direction} | {row['goal']} | "
                     f"{row['contents']} | {row['layers']} / {row['width']} / {row['state_size']} | "
                     f"{row['window']} / {row['lags']} | {row['batch_size']} | {len(row['runs'])} |")
    lines += ["", f"Total: **{sum(len(r['runs']) for r in rows)} S4-family runs** across {len(rows)} distinct experiment/variant/window/lag configurations.",
              "", "## Shared settings and interpretation", "",
              "- All backbones: four layers, width 128, state size 64, `n_ssm=1`, dropout 0.1, pre-norm residual blocks and final LayerNorm. Width/depth matching does not imply equal parameter counts.",
              "- 1-D S4: `s4` is causal, `s4_bidir` is bidirectional within the observed history. Both use the authors' temporal DPLR/FFT kernel, rank 1, HiPPO-LegS initialization, `dt_transform=softplus`, GELU/GLU and a last-time-step linear head. Directions have separate identities, output directories and checkpoints; parameter counts can differ.",
              "- S4ND: bidirectional within the observed input grid, separable DPLR kernels, tensor rank 1; padded cells are excluded from layers and loss. No future target observations enter the input.",
              "- `s4nd_2d_state` genuinely convolves over two axes. The historical `s4nd_2d` label actually uses three convolution axes (state, group, time); it is not identical to the state-only version.",
              "- Axial attention: one single-head attention module per selected categorical axis, attention dropout 0.0, followed by four modular temporal S4 layers.",
              "- Training: AdamW, fixed initial LR 0.0001 (no LR sweep), weight decay 0.0001 with author-marked SSM exemptions, betas (0.9, 0.999), epsilon 1e-8; no private kernel LR override. The existing plateau scheduler is retained.",
              "- Selection: normalized MSE on value/weight, 5000-epoch safety cap, validation every epoch, early stopping patience 50 and minimum improvement 0.0001. Plateau scheduler patience 15, factor 0.5; gradient clipping 1.0; FP32, AMP/TF32 off.",
              "- Exp 1 aggregates the eight-state panel to one series: batch 33, 132 training windows, four updates/epoch. Category IDs are constant in the aggregate arms.",
              "- Exp 2 uses one full target month per update: 918 valid lanes, 132 updates/epoch. Flat batch 918, axial/grid batch 1, state-only S4ND batch 172 groups, S4ND 3D batch 2 flows are different layouts of that same month.",
              "- Exp 3 uses one full target month per update, batch 918, common training start 2015-01, 108 updates/epoch. Every combination of windows/lags {6,12,24,36} is included.",
              "- Train-only cohort and MinMax fit; train ends 2023-12, validation is 2024-01 through 2024-12, test is 2025-01 through 2025-05. All experiments predict the next month; window/lag values are not forecast horizons.",
              "- `matched_axial_2d/3d/4d` are tested aliases of the corresponding `cross_attention_*` models, not extra paper runs.", ""]
    for experiment in (1, 2, 3):
        seeds = sorted({r["seed"] for row in rows if row["experiment"] == experiment for r in row["runs"]})
        lines.append(f"Exp {experiment} seeds: " + ", ".join(map(str, seeds)) + ".")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    rows = inventory()
    print(json.dumps(rows, indent=2) if args.format == "json" else markdown(rows), end="\n")


if __name__ == "__main__":
    main()
