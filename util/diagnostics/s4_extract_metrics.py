"""Extract final normalized MSE from S4 train.log into metrics.json.

S4 runs (PyTorch Lightning) only ever log loss on MinMax-normalized (0-1)
values. We surface that as `norm_mse_combined` -- the same quantity the
github/baseline models now report under --normalized (mean MSE over the
flattened value+weight targets). This makes every model comparable on one
scale.

CAVEAT -- the log number is NOT the canonical metric. `final/test/loss` in the
Lightning log comes from `trainer.test(model)` called right after fit with no
ckpt_path, i.e. it evaluates the LAST-EPOCH weights. The canonical protocol
evaluates the best-val checkpoint (`hydra/checkpoints/val/loss.ckpt`) via
scripts/s4_eval.py, which also writes the full metric suite, the prediction
artifacts, and cohort provenance. This extractor is therefore a smoke-check
fallback only:

  * by default it REFUSES to write anything, prints the last-epoch number it
    found so a smoke-check still gets an answer, and points at s4_eval.py;
  * with --allow-last-epoch it writes a payload whose mode is
    "one_step_normalized_last_epoch" plus an explicit note, so provenance
    gates (src/utils/provenance.py requires mode "one_step_normalized" and
    cohort provenance) reject it as a completion marker and it can never be
    mistaken for a corrected-protocol number.

The log line also carries no cohort and no target count, so nothing built
from it can claim the corrected train-only cohort (918 combos / 4,590 test
targets) in the first place.

Exit status is a completeness signal over EVERY s4_* run in the directory,
not just the ones this script looked at writing: it is 0 only when every run
already carries a metrics.json that passes the same provenance gate
collect_results.py applies. Grid runs, runs holding a legacy metrics.json
from an older version of this script, and runs this invocation just wrote a
last-epoch payload for all still count as incomplete, because for each of
them the canonical artifact is still missing.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.provenance import (
    corrected_per_combo_artifact,
    corrected_prediction_artifact,
)


def parse_train_log(log_path):
    test_loss = val_loss = None
    for line in log_path.read_text().splitlines():
        line = line.replace("│", "|").strip()
        if "final/test/loss" in line:
            m = re.search(r"([\d.][\d.eE+\-]*)", line.split("final/test/loss")[-1])
            if m:
                test_loss = float(m.group(1))
        if "final/val/loss" in line:
            m = re.search(r"([\d.][\d.eE+\-]*)", line.split("final/val/loss")[-1])
            if m:
                val_loss = float(m.group(1))
    return test_loss, val_loss


def canonical_status(run_dir):
    """Does this run already hold a canonical, corrected-protocol metrics.json?

    Deliberately the SAME gate collect_results.py and build_paper_results.py
    apply, so "this script exited 0" and "the collectors will accept these
    runs" cannot drift apart. A metrics.json existing is not enough: the
    payload this script used to write (and any pre-correction 903-cohort
    artifact) has a filename but no valid cohort provenance.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return False, "no metrics.json"
    try:
        payload = json.loads(metrics_path.read_text())
    except (OSError, ValueError) as exc:
        return False, f"unreadable metrics.json ({exc})"
    if not isinstance(payload, dict):
        return False, "metrics.json is not an object"
    ok, reason = corrected_per_combo_artifact(payload)
    if not ok:
        return False, reason
    return corrected_prediction_artifact(payload, metrics_path)


def main():
    ap = argparse.ArgumentParser(
        description="Fallback: surface S4 train.log losses as metrics.json "
                    "(last-epoch weights, NON-canonical -- see module docstring)."
    )
    ap.add_argument("runs_dir", nargs="?", default="outputs/full", type=Path)
    ap.add_argument(
        "--allow-last-epoch", action="store_true",
        help="Actually write metrics.json from the log. The payload is marked "
             "mode=one_step_normalized_last_epoch so provenance gates reject "
             "it; the canonical path is scripts/s4_eval.py (best-val weights).",
    )
    args = ap.parse_args()

    if not args.runs_dir.is_dir():
        print(f"[error] no such runs dir: {args.runs_dir}")
        return 2

    incomplete = []
    for d in sorted(args.runs_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("s4_"):
            continue
        canonical, reason = canonical_status(d)
        if canonical:
            print(f"[ok] {d.name}: canonical metrics.json present")
            continue
        incomplete.append([d.name, reason])
        # Grid (S4ND) runs: the Lightning final/test/loss is the UNMASKED dense-
        # grid MSE — it averages over the ~40%+ structurally-absent zero-padded
        # cells, so it is NOT the kept-combo metric the github models report.
        # s4_eval.py computes the masked value for these; never publish the log
        # number for a grid run.
        if "s4nd" in d.name:
            print(f"[skip] {d.name}: grid run ({reason}) — metric must come "
                  f"from s4_eval.py (masked)")
            continue
        log = d / "train.log"
        out = d / "metrics.json"
        if not log.exists():
            print(f"[warn] {d.name}: {reason}, and no train.log to fall back on")
            continue
        if out.exists():
            # Present but not canonical — a legacy payload from an older
            # version of this script, or a pre-correction artifact. Never
            # overwritten here; s4_eval.py replaces it with the real thing.
            print(f"[warn] {d.name}: metrics.json is NOT canonical ({reason}); "
                  f"rerun scripts/s4_eval.py to replace it")
            continue
        test_mse, val_mse = parse_train_log(log)
        if test_mse is None:
            print(f"[warn] {d.name}: no final/test/loss in train.log (failed run?)")
            continue
        if not args.allow_last_epoch:
            print(f"[refuse] {d.name}: train.log holds a LAST-EPOCH test loss "
                  f"({test_mse:.6f}, val={val_mse}), not the best-val-checkpoint "
                  f"metric. Run scripts/s4_eval.py for the canonical "
                  f"metrics.json, or re-run with --allow-last-epoch to write a "
                  f"clearly-labeled non-canonical payload.")
            continue
        # Parse the seed from the directory name — it was hardcoded to 42,
        # so a fallback firing for a seed-26/105/115 run would mislabel the
        # payload seed AND leave "_seed<N>" inside the variant string,
        # splitting the arm in two in build_paper_results.py.
        seed_m = re.search(r"_seed(\d+)$", d.name)
        run_seed = int(seed_m.group(1)) if seed_m else 42
        variant = re.sub(r"_seed\d+$", "", d.name[len("s4_"):])
        payload = {
            "model": "s4",
            "variant": variant,
            # NOT "one_step_normalized": provenance gates must reject this.
            "mode": "one_step_normalized_last_epoch",
            "seed": run_seed,
            "note": "NON-CANONICAL: normalized (0-1) MSE parsed from the S4 "
                    "Lightning log's final/test/loss, which evaluates the "
                    "LAST-EPOCH weights (trainer.test after fit, no ckpt_path). "
                    "The canonical number is s4_eval.py's evaluation of the "
                    "best-val checkpoint.",
            "metrics": {
                "norm_mse_combined": test_mse,
                "norm_val_mse": val_mse,
            },
        }
        out.write_text(json.dumps(payload, indent=2))
        # Written, but still counted as incomplete above: a last-epoch payload
        # is a smoke-check, not the canonical artifact.
        incomplete[-1][1] = "last-epoch payload written; canonical still missing"
        print(f"[wrote] {d.name}: norm_mse_combined={test_mse:.6f} "
              f"(val={val_mse}) [last-epoch, non-canonical]")

    if incomplete:
        print(f"\n{len(incomplete)} run(s) lack a canonical metrics.json:")
        for name, reason in incomplete:
            print(f"    {name}: {reason}")
        print("Produce it with:\n"
              "    cd external/s4 && python ../../scripts/s4_eval.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
