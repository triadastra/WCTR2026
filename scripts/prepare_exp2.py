#!/usr/bin/env python
"""Inspect and validate Experiment 2 before training; never starts training."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contextlib
with contextlib.redirect_stdout(sys.stderr):
    from src.utils.exp2_protocol import (
        comparable_s4_contract, expected_runs, expected_contract, s4_overrides, validate_contract,
    )
    from src.utils.experiment_manifest import load_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT)
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--check-runtime", action="store_true")
    ap.add_argument("--check-results", action="store_true")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/full")
    ap.add_argument("--s4-overrides", action="store_true")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--native-checkpoint", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--variant")
    ap.add_argument("--s4-checkpoint", type=Path)
    ap.add_argument("--experiment")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.native_checkpoint:
        import torch
        manifest = load_manifest(ROOT / "config/experiment_manifest.env")
        runs = [r for r in expected_runs(manifest, [args.seed])
                if r.kind == "native" and (r.model, r.variant) == (args.model, args.variant)]
        if len(runs) != 1 or args.seed is None:
            ap.error("--native-checkpoint requires --model, --variant and --seed")
        try:
            checkpoint = torch.load(args.native_checkpoint, map_location="cpu", weights_only=False)
            if checkpoint.get("run_contract") != expected_contract(ROOT, runs[0], manifest):
                raise ValueError("saved training recipe differs from current recipe")
        except Exception as exc:
            print(f"[stale checkpoint] {exc}", file=sys.stderr)
            return 1
        return 0
    if args.s4_checkpoint:
        import torch
        manifest = load_manifest(ROOT / "config/experiment_manifest.env")
        runs = [r for r in expected_runs(manifest, [args.seed]) if r.kind == "s4" and r.experiment == args.experiment]
        if len(runs) != 1 or args.seed is None:
            ap.error("--s4-checkpoint requires --experiment and --seed")
        try:
            checkpoint = torch.load(args.s4_checkpoint, map_location="cpu", weights_only=False)
            if (comparable_s4_contract(checkpoint.get("run_contract"))
                    != comparable_s4_contract(expected_contract(ROOT, runs[0], manifest))):
                raise ValueError("saved S4 recipe differs from current recipe")
        except Exception as exc:
            print(f"[stale checkpoint] {exc}", file=sys.stderr)
            return 1
        return 0
    if args.s4_overrides:
        print("\n".join(s4_overrides(ROOT)))
        return 0
    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    runs = expected_runs(manifest)
    report = {"experiment": 2, "arms": len(runs) // len(manifest["PAPER_SEEDS"].split()),
              "seeds": list(map(int, manifest["PAPER_SEEDS"].split())), "runs": len(runs),
              "errors": [], "checks": {}, "run_directories": [r.directory_name for r in runs]}
    try:
        # Resolve every recipe, including all six Hydra architectures, without
        # importing GPU kernels or reading test observations.
        for run in runs:
            expected_contract(ROOT, run, manifest)
        report["checks"]["contracts"] = "passed"
    except Exception as exc:
        report["errors"].append(f"contracts: {exc}")
    if args.check_runtime:
        from src.utils.runtime_readiness import check_runtime
        runtime = check_runtime(ROOT)
        report["checks"]["runtime"] = runtime
        report["errors"].extend("runtime unavailable: " + error for error in runtime["errors"])
    if args.check_data:
        try:
            from src.data.dataloader import TradeDataPipeline, DataConfig
            from src.utils.config import load_config, get_data_config
            cfg = get_data_config(load_config(str(ROOT / "config/base.yaml")))
            cfg.update(imports_dir=str(args.data_dir.resolve() / "data/imports"),
                       exports_dir=str(args.data_dir.resolve() / "data/exports"))
            from src.utils.config import resolve_data_paths
            cfg = resolve_data_paths(cfg, args.data_dir.resolve())
            pipe = TradeDataPipeline(DataConfig(**cfg))
            pipe.load_data()
            pipe.create_splits()
            n = len(pipe.kept_combos)
            flat, val, test = pipe.get_dataloaders(batch_size=n, num_workers=0, shuffle_train=False)
            combo, combo_val, combo_test = pipe.get_dataloaders(batch_size=1, num_workers=0, combo=True, shuffle_train=False)
            flat_keys = {((s["state_id"], s["comm_id"], s["flow_id"]), str(s["target_time"])) for s in flat.dataset.samples}
            # All dense cohort series must have every common target month.
            months = {str(s["target_time"]) for s in flat.dataset.samples}
            assert len(flat_keys) == n * len(months) == len(flat.dataset)
            assert len(combo.dataset) == len(months)
            assert len(flat) == len(combo) == len(months)
            assert len(val.dataset) == n * len(combo_val.dataset)
            assert len(test.dataset) == n * len(combo_test.dataset)
            from src.utils.provenance import CORRECTED_NUM_COMBOS, CORRECTED_TEST_TARGETS
            assert n == CORRECTED_NUM_COMBOS and len(test.dataset) == CORRECTED_TEST_TARGETS
            base = load_config(str(ROOT / "config/base.yaml"))
            assert base["data"]["batch_by_target_month"]
            assert pipe.config.cohort_from_train_only and pipe.config.tabular_align_targets
            from src.utils.experiment_manifest import exp1_source_domain
            assert pipe.source_domain == exp1_source_domain(manifest)
            report["checks"]["data"] = {"source_fingerprint": pipe.source_fingerprint,
                "target_measure": pipe.config.target_measure, "duplicate_policy": pipe.config.duplicate_policy,
                "combos": n, "train_targets": len(flat.dataset), "validation_targets": len(val.dataset),
                "test_targets": len(test.dataset), "steps_per_epoch": len(flat),
                "first_train_target": min(months), "last_train_target": max(months)}
        except Exception as exc:
            report["errors"].append(f"data: {type(exc).__name__}: {exc}")
    if args.check_results:
        from src.utils.provenance import corrected_per_combo_artifact, corrected_prediction_artifact, prediction_truth_digest
        roster = {r.directory_name: r for r in runs}
        expected = set(roster)
        truth_digest = None
        actual = {p.parent.name for p in args.runs_dir.glob("*/metrics.json")}
        for name in sorted(expected - actual):
            report["errors"].append(f"missing result: {name}")
        for name in sorted(actual - expected):
            report["errors"].append(f"unexpected result: {name}")
        for name in sorted(actual & expected):
            path = args.runs_dir / name / "metrics.json"
            try:
                payload = json.loads(path.read_text())
                run = roster[name]
                if any(payload.get(key) != value for key, value in (
                    ("model", run.model), ("variant", run.variant), ("seed", run.seed)
                )):
                    raise ValueError("directory/manifest identity mismatch")
                for check in (lambda: corrected_per_combo_artifact(payload),
                              lambda: validate_contract(payload, ROOT),
                              lambda: corrected_prediction_artifact(payload, path, verify_scaled_metrics=True)):
                    valid, reason = check()
                    if not valid:
                        raise ValueError(reason)
                current_digest = prediction_truth_digest(payload, path)
                from src.utils.exp2_completion import validate_exp2_completion_artifact
                validate_exp2_completion_artifact(payload, path)
                if truth_digest is not None and current_digest != truth_digest:
                    raise ValueError("test targets differ across runs")
                truth_digest = current_digest
            except Exception as exc:
                report["errors"].append(f"{name}: {exc}")
        report["checks"]["results"] = {"expected": len(expected), "found": len(actual)}
    report["checks_passed"] = not report["errors"]
    report["ready_to_train"] = args.check_data and args.check_runtime and not report["errors"]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "run_directories"}, indent=2))
    return 0 if report["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
