#!/usr/bin/env python
"""Exit successfully only for a current, corrected metrics artifact."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.provenance import (
    CORRECTED_TEST_DATES,
    CORRECTED_TEST_MONTHS,
    corrected_aggregate_artifact,
    corrected_per_combo_artifact,
    corrected_prediction_artifact,
)
from src.utils.experiment_manifest import (
    exp1_expected_runs,
    exp1_protocol,
    exp1_source_domain,
    load_manifest,
)
from src.utils.run_contract import expected_exp1_run_contract


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--input-len", type=int)
    ap.add_argument("--lag-count", type=int)
    # argparse.SUPPRESS distinguishes "caller did not constrain this field"
    # from the canonical null value used by Exp 1.  The former code used None
    # for both, so an artifact trained on a shortened target history could be
    # accepted as a current Exp 1 run.
    ap.add_argument("--train-target-start", default=argparse.SUPPRESS)
    ap.add_argument("--expected-seed", type=int)
    ap.add_argument("--expected-model")
    ap.add_argument("--expected-variant")
    ap.add_argument(
        "--exp1-manifest", type=Path,
        help="Require the complete canonical Exp 1 source/protocol contract.",
    )
    ap.add_argument("--exp3", action="store_true", help="Require the full canonical Exp 3 contract")
    ap.add_argument("--exp2", action="store_true", help="Require the canonical Experiment 2 recipe")
    ap.add_argument(
        "--linhead", action="store_true",
        help="Check against the linear-head Transformer/GPT rerun roster "
             "(src/utils/linhead_protocol.py) instead of the paper roster",
    )
    args = ap.parse_args()
    if not args.metrics.is_file():
        return 1
    try:
        payload = json.loads(args.metrics.read_text())
        current, reason = (
            corrected_aggregate_artifact(payload) if args.aggregate
            else corrected_per_combo_artifact(payload)
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"[stale] {args.metrics}: unreadable metrics ({exc})")
        return 1
    if not current:
        print(f"[stale] {args.metrics}: {reason}")
        return 1
    if args.exp1_manifest is not None:
        if not args.aggregate:
            print(f"[stale] {args.metrics}: --exp1-manifest requires --aggregate")
            return 1
        try:
            manifest = load_manifest(args.exp1_manifest)
            canonical = exp1_protocol(manifest)
            canonical_domain = exp1_source_domain(manifest)
        except (OSError, ValueError, TypeError) as exc:
            print(f"[stale] {args.metrics}: invalid Exp 1 manifest ({exc})")
            return 1
        if None in (args.expected_seed, args.expected_model, args.expected_variant):
            print(
                f"[stale] {args.metrics}: --exp1-manifest requires the expected "
                "model, variant, and seed"
            )
            return 1
        if args.linhead:
            from src.utils.linhead_protocol import exp1_runs as roster
        else:
            roster = exp1_expected_runs
        matching_runs = [
            run for run in roster(manifest)
            if (
                run.model == args.expected_model
                and run.variant == args.expected_variant
                and run.seed == args.expected_seed
            )
        ]
        if len(matching_runs) != 1:
            print(f"[stale] {args.metrics}: requested run is not in the "
                  f"{'linear-head' if args.linhead else 'Exp 1'} manifest")
            return 1
        if payload.get("source_fingerprint") != canonical["source_fingerprint"]:
            print(f"[stale] {args.metrics}: source fingerprint is not canonical")
            return 1
        if payload.get("source_domain") != canonical_domain:
            print(f"[stale] {args.metrics}: source population/measure/calendar is not canonical")
            return 1
        actual_protocol = payload.get("data_protocol", {})
        for key in ("input_len", "lag_count", "train_target_start"):
            if actual_protocol.get(key) != canonical[key]:
                print(
                    f"[stale] {args.metrics}: {key}={actual_protocol.get(key)!r}, "
                    f"manifest requires {canonical[key]!r}"
                )
                return 1
        expected_window = (
            canonical["moving_average_window"]
            if payload.get("model") == "moving_average" else None
        )
        if payload.get("model") in {
            "seasonal_naive", "random_walk", "moving_average"
        } and payload.get("window") != expected_window:
            print(
                f"[stale] {args.metrics}: window={payload.get('window')!r}, "
                f"manifest requires {expected_window!r}"
            )
            return 1
        repo = Path(__file__).resolve().parents[1]
        expected_contract = expected_exp1_run_contract(
            repo, matching_runs[0], canonical, canonical_domain
        )
        if payload.get("run_contract") != expected_contract:
            print(
                f"[stale] {args.metrics}: embedded run contract does not match "
                "the canonical Exp 1 model/data/training/selection contract"
            )
            return 1
        from src.utils.exp1_completion import validate_exp1_completion_artifact
        try:
            validate_exp1_completion_artifact(payload, args.metrics)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[stale] {args.metrics}: {exc}")
            return 1
    if args.exp3:
        from src.utils.exp3_protocol import validate_contract
        current, reason = validate_contract(payload, args.expected_model, args.input_len,
                                            args.lag_count, args.expected_seed)
        if args.aggregate or not current:
            print(f"[stale] {args.metrics}: {reason}")
            return 1
    if args.exp2:
        from src.utils.exp2_protocol import validate_contract
        roster = None
        if args.linhead:
            from src.utils.linhead_protocol import exp2_runs as roster
        current, reason = validate_contract(payload, Path(__file__).resolve().parents[1], roster)
        if not current:
            print(f"[stale] {args.metrics}: {reason}")
            return 1
        from src.utils.exp2_completion import validate_exp2_completion_artifact
        try:
            validate_exp2_completion_artifact(payload, args.metrics)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[stale] {args.metrics}: {exc}")
            return 1
    if args.exp3:
        from src.utils.training_completion import validate_completion_artifact
        try:
            validate_completion_artifact(payload, args.metrics)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[stale] {args.metrics}: {exc}")
            return 1
    identity = {
        "seed": args.expected_seed,
        "model": args.expected_model,
        "variant": args.expected_variant,
    }
    for key, expected_value in identity.items():
        actual_value = payload.get(key)
        if expected_value is not None and (
            actual_value != expected_value or type(actual_value) is not type(expected_value)
        ):
            print(
                f"[stale] {args.metrics}: {key}={payload.get(key)!r}, "
                f"expected {expected_value!r}"
            )
            return 1
    expected = {}
    if args.input_len is not None:
        expected["input_len"] = args.input_len
    if args.lag_count is not None:
        expected["lag_count"] = args.lag_count
    if hasattr(args, "train_target_start"):
        expected["train_target_start"] = (
            None if args.train_target_start == "null" else args.train_target_start
        )
    protocol = payload["data_protocol"]
    for key, value in expected.items():
        # Presence in ``expected`` is the constraint. In particular, an
        # explicit --train-target-start null must require an actual null.
        if protocol.get(key) != value:
            print(f"[stale] {args.metrics}: {key}={protocol.get(key)!r}, expected {value!r}")
            return 1
    prediction_kwargs = (
        {
            "expected_num_combos": 1,
            "expected_months": CORRECTED_TEST_MONTHS,
            "expected_dates": CORRECTED_TEST_DATES,
            "expected_combos": {("ALL", "ALL", "ALL")},
        }
        if args.aggregate else {}
    )
    current, reason = corrected_prediction_artifact(
        payload, args.metrics, verify_scaled_metrics=args.exp2 or args.exp3, **prediction_kwargs
    )
    if not current:
        print(f"[stale] {args.metrics}: {reason}")
        return 1
    if args.exp1_manifest is not None:
        from src.utils.exp1_truth import load_exp1_truth, validate_exp1_truth
        try:
            repo = Path(__file__).resolve().parents[1]
            truth = load_exp1_truth(repo, Path(os.environ.get("DATA_DIR", str(repo))))
            current, reason = validate_exp1_truth(payload, args.metrics, truth)
        except (OSError, ValueError, TypeError) as exc:
            current, reason = False, f"pinned source validation failed: {exc}"
        if not current:
            print(f"[stale] {args.metrics}: {reason}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
