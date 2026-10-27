"""Linear-head rerun of the Exp 1/2 Transformer and GPT arms.

The published Transformer (128 -> 64 -> 2) and GPT (128 -> 128 -> 2) arms read
out through an MLP while every other neural arm uses Linear(128 -> 2). This
rerun trains the same arms with the linear head under the model names
``transformer_linhead`` / ``gpt_linhead`` (config/models/*_linhead.yaml), so
they never share a directory, results row or SHAP table with the published
runs. The roster is derived from the canonical manifest: every Exp 1/2 run of
``transformer`` or ``gpt`` gets exactly one linear-head twin with the same
variant and seed, and nothing else. Data, training recipe and selection are
the canonical ones; only the head differs.

Outputs live under ``outputs/linhead/{agg,full}`` because the paper gate in
scripts/run_all_experiments.sh rejects any unexpected directory under
``outputs/agg`` or ``outputs/full``.
"""
from pathlib import Path

from .experiment_manifest import Exp1Run, exp1_expected_runs, load_manifest
from .exp2_protocol import Exp2Run, expected_runs as exp2_expected_runs

ROOT = Path(__file__).resolve().parents[2]
LINHEAD_MODELS = {"transformer": "transformer_linhead", "gpt": "gpt_linhead"}
EXP1_DIR = "outputs/linhead/agg"
EXP2_DIR = "outputs/linhead/full"


def manifest():
    return load_manifest(ROOT / "config/experiment_manifest.env")


def exp1_runs(m=None):
    m = m or manifest()
    return tuple(sorted(
        Exp1Run(LINHEAD_MODELS[r.model], r.variant, r.seed, r.kind)
        for r in exp1_expected_runs(m) if r.model in LINHEAD_MODELS
    ))


def exp2_runs(m=None, seeds=None):
    m = m or manifest()
    return tuple(
        Exp2Run(LINHEAD_MODELS[r.model], r.variant, r.seed, r.kind, r.experiment)
        for r in exp2_expected_runs(m, seeds)
        if r.kind == "native" and r.model in LINHEAD_MODELS
    )
