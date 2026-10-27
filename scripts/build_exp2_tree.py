#!/usr/bin/env python
"""Curate graphs/exp2_benchmark/ from the per-run plots inference.py writes.

This step existed only as an ad-hoc script that was never committed, so the
tree could not be rebuilt from the repo -- after the post-correction sweep it
still held figures from 2026-06-22, showing pre-correction runs under
current-looking filenames, with no figure at all for arms added since. That is
worse than a missing tree: it silently contradicts results.xlsx.

Inputs are what `scripts/inference.py` already produces per run:

    outputs/full/<model>_<variant>_seed<N>/plots/
        scatter_value.png, scatter_weight.png, loss_curve.png
        combos/<idx>_<state>_<commodity>_<flow>_<target>.png

Output is one folder per arm, seeds collapsed:

    graphs/exp2_benchmark/scatter/<arm>.png       (predicted vs actual)
    graphs/exp2_benchmark/timeline/<arm>/*.png    (per-combo test horizon)

Seeds are collapsed by taking the LOWEST seed present, not by averaging: these
are illustrative plots of individual forecasts, and averaging pixel data across
seeds would show a curve no model actually produced. The metric tables are where
seed-averaging belongs. The chosen seed is recorded in the folder's SEED file so
a figure in the paper can always be traced back to the run that made it.
"""

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "full"
OUT = ROOT / "graphs" / "exp2_benchmark"

_RUN = re.compile(r"^(?P<arm>.+)_seed(?P<seed>\d+)$")


def discover():
    """arm -> (lowest seed, run dir). Skips runs with no plots directory."""
    best = {}
    for d in sorted(RUNS.glob("*_seed*")):
        m = _RUN.match(d.name)
        if not m or not (d / "plots").is_dir():
            continue
        arm, seed = m.group("arm"), int(m.group("seed"))
        if arm not in best or seed < best[arm][0]:
            best[arm] = (seed, d)
    return best


def main():
    if not RUNS.exists():
        print(f"[exp2] runs dir not found: {RUNS}")
        return 1

    arms = discover()
    if not arms:
        print(f"[exp2] no runs with plots under {RUNS}")
        return 1

    # Rebuild rather than merge: a stale figure left behind from a superseded
    # arm is exactly the failure this script exists to prevent.
    for sub in ("scatter", "timeline"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    (OUT / "scatter").mkdir(parents=True, exist_ok=True)
    (OUT / "timeline").mkdir(parents=True, exist_ok=True)

    n_scatter = n_timeline = 0
    for arm, (seed, d) in sorted(arms.items()):
        src = d / "plots" / "scatter_value.png"
        if src.exists():
            shutil.copy2(src, OUT / "scatter" / f"{arm}.png")
            n_scatter += 1

        combos = d / "plots" / "combos"
        if combos.is_dir():
            dest = OUT / "timeline" / arm
            dest.mkdir(parents=True, exist_ok=True)
            # Value-target plots only; weight duplicates the layout and doubles
            # the tree for no extra information at a glance.
            picked = sorted(p for p in combos.glob("*_value.png"))
            for p in picked:
                shutil.copy2(p, dest / p.name)
            if picked:
                (dest / "SEED").write_text(f"{d.name}\n")
                n_timeline += 1

    print(f"[exp2] {len(arms)} arms | {n_scatter} scatter | {n_timeline} timeline folders")
    print(f"[exp2] seeds used: {', '.join(f'{a}={s}' for a, (s, _) in sorted(arms.items())[:3])} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
