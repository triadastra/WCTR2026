"""The fixed Exp 2 comparison family, including separate S4 directions.

One declared list, consumed by every statistical script, so that a table or
figure can only be built from the COMPLETE family -- never from whatever run
directories happen to exist. Mirrors the tiers of scripts/run_full_test.sh
(pinned by tests/test_statistical_consumers.py) and the manifest check in
scripts/run_all_experiments.sh::verify_paper_outputs.

Directory naming: the github-side arms are ``<model>_<variant>_seed<N>``; the
hydra S4 family is evaluated by scripts/s4_eval.py into ``s4_<exp>_seed<N>``
where the payload records model ``s4`` / ``s4nd`` and the bare variant.
"""
from __future__ import annotations

SEEDS = (26, 105, 115)

from pathlib import Path
from .experiment_manifest import load_manifest
from .exp2_protocol import expected_runs as protocol_runs
_MANIFEST = load_manifest(Path(__file__).resolve().parents[2] / "config/experiment_manifest.env")
EXP2_ARMS = tuple((run.model, run.variant) for run in protocol_runs(_MANIFEST, seeds=[26]))

# Hydra-evaluated arms live under the s4_ prefix (scripts/s4_eval.py).
_RUN_TEMPLATES = {(run.model, run.variant): run for run in protocol_runs(_MANIFEST, seeds=[26])}


def arm_key(model: str, variant: str) -> str:
    return f"{model}_{variant}"


def run_dir_name(model: str, variant: str, seed: int) -> str:
    from dataclasses import replace
    return replace(_RUN_TEMPLATES[(model, variant)], seed=seed).directory_name


def expected_runs(seeds=SEEDS) -> dict[str, tuple[str, str, int]]:
    """directory name -> (model, variant, seed) for the whole family."""
    return {run_dir_name(m, v, s): (m, v, s) for m, v in EXP2_ARMS for s in seeds}


def check_roster(found_dirs, seeds=SEEDS) -> dict[str, tuple[str, str, int]]:
    """Refuse anything but the exact family; return the directory -> identity map."""
    expected = expected_runs(seeds)
    found = set(found_dirs)
    missing = sorted(set(expected) - found)
    unexpected = sorted(found - set(expected))
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)}: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            detail.append(f"unexpected {len(unexpected)}: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
        raise RuntimeError(
            "Inexact Exp 2 roster under outputs/full (" + "; ".join(detail) +
            f"). Statistics require the complete {len(EXP2_ARMS)}-arm x {len(seeds)}-seed family."
        )
    return expected
