# Experiment 3 review — 2026-09-05

> Historical audit snapshot. The current manifest adds `s4_bidir`: four model
> families × 16 window/lag cells × three seeds = 192 runs. See the
> [results snapshot](../results/2026-10-26/README.md) for the completed sweep; the dated
> findings below retain their original context.

## Outcome

The current sweep is a window/lag sensitivity experiment: LSTM, Transformer,
and S4; embedding encoding; windows and lag counts 6, 12, 24, 36; seeds
26, 105, 115 (144 runs). No existing `outputs/sweep` results were available on
this machine to certify historical scores. This review covers the current
code, configuration, and preserved input files.

## Findings and fixes

1. **High: off-protocol training could enter the sweep.** The skip gate and
   sweep collector checked history coordinates and general corrected-run
   provenance but did not reconstruct the complete expected training contract.
   Different batch sizes, learning rates, epoch limits, architectures, or
   selection policies could therefore be accepted under the same cell name.
   `src/utils/exp3_protocol.py` now reconstructs the expected contract; the
   master runner, standalone runner, skip checker, and collector enforce it.
   Payload identity must also match the directory's model and seed.
2. **High: heatmaps bypassed provenance.** The old plotting path read scalar
   MSEs directly from JSON, even when all three runs were stale. It now uses
   the paper collector's complete-grid, contract, and prediction validation.
3. **Medium: standalone failure/readiness handling.** The old shell runner
   forced a machine-specific Conda environment and data directory, could leave
   stale completion markers during retraining, and exited successfully despite
   failed cells. It now delegates to the portable Python controller, preflights
   inputs, fails on errors, suppresses inference after training failure, and
   shares the master controller's lock. Successfully validated cells are skipped;
   incomplete/stale cells restart. It does not resume partial checkpoints.
4. **Source limitation, retained explicitly.** The live shared configuration
   preserves the original source snapshot: import `Vessel Value ($US)` /
   `Vessel SWT (kg)` and containerized-vessel exports, and all duplicate records summed. Preflight observed
   2,250 duplicate source-key groups / 4,500 involved rows, including 2,198
   groups with conflicting selected targets. These records were not changed.
   This supports comparisons on the defined original data, not a claim of
   independently validated or duplicate-free trade totals. See
   `docs/LEGACY_DATA_AUDIT.md`. Shared configuration was updated during this
   review; Exp 3 follows its current manifest rather than replacing it.

## Leakage and interpretation

- Current cohort selection uses only data through December 2023: 918 series.
  Full-timeline 903-series historical selection is holdout-dependent and must
  not be mixed with the corrected sweep.
- Min/max scaling is fit on training data only and then applied to holdouts.
  Lag features are shifted backward; input windows end before their target.
- All cells start supervised training targets in January 2015 and end in
  December 2023 (108 target months). Earlier data provide history and shared
  train-only preprocessing. Validation covers 2024; test covers Jan–May 2025.
- Evaluation is rolling one-month-ahead with observed previous-month values.
  It is not a five-month forecast made entirely in December 2024. Operational
  use also needs a source-release-delay assumption; these files do not verify
  that each previous month's data were available at the forecast date.
- Checkpoint selection uses validation MSE. Test heatmaps are descriptive:
  selecting the best window/lag from them and reporting the same test score as
  an unbiased tuned-model result would introduce test-selection bias. Select
  settings using validation or a separate temporal tuning study first.
- Equal width/depth and optimizer settings do not make these architectures
  parameter- or compute-matched. Changing lag count changes feature dimension;
  history reach is window + lag (12–72 months). Report model size and avoid
  attributing differences solely to memory length. This is a fixed-protocol
  sensitivity comparison, not an equal-budget hyperparameter optimization study.
- Five test months support limited temporal generalization claims. Three seeds
  measure initialization variability, not uncertainty across future years.

## Runbook

Run from the repository root with the project Python dependencies installed:

```bash
python scripts/run_exp3.py plan
python scripts/run_exp3.py check --data-dir .
CUDA_VISIBLE_DEVICES=0 python scripts/run_exp3.py train --data-dir .
python scripts/run_exp3.py collect
python util/build_sweep_heatmaps.py
```

The shell compatibility entry point is `bash util/run_exp3_sweep.sh train`.
Use `PY=/path/to/python` to choose its environment. Configuration comes from
`config/experiment_manifest.env`, base YAML, and model/variant YAML. The
standalone sweep is sequential; do not run another workload on its GPU when
comparing runtime. The master controller still supports multiple GPUs, so its
runs do not establish a controlled same-device speed comparison.

`collect` writes `outputs/sweep/exp3_cells.csv` only after all canonical runs
pass. The paper workbook can subsequently be built with the existing master
pipeline once the other experiments are also complete. If an interrupted
process leaves `outputs/.run_all_experiments.lock`, confirm no controller is
running before removing the empty lock directory.

No full training or benchmark results were produced in this review. Local
PyTorch is CPU-only for this purpose (`torch.cuda.is_available()` is false).

## Verification

- Read-only preflight passed on the preserved original source snapshot and
  matched the manifest's source domain and duplicate audit: 918 retained series,
  12 validation months, five test months; CUDA unavailable locally.
- All 18 new Exp 3 tests passed, including contract mutations, relabelled seeds,
  heatmap gate propagation, launcher failure suppression, and finite forward /
  backward passes for all three architectures at (6,6) and (36,36).
- All seven real-source data-pipeline tests passed, covering train-only cohort,
  train-only scaling, and 4,590 test targets. Their fixture now reads the pinned
  YAML source profile rather than strict bare DataConfig defaults.
- Eight burn-in/window tests passed, including aligned short/long Exp 3 targets.
- Shell syntax and Python compilation checks passed. No raw CSVs were modified.

## Follow-up: completion and prediction-head policy

Experiment 3 now requires a completed early-stopping run before test inference,
reuse, or publication. The trainer writes an atomic
`artifacts/training_completion.json` only after a successful training return
path. It records the stopping reason, epochs, best validation score/epoch,
patience counter, full run contract, and SHA-256 of `best.pth`. Retraining clears
old evidence after resume compatibility checks. Missing, interrupted,
epoch-capped, mismatched, or checkpoint-tampered evidence fails the Exp 3 gate.
Inference embeds the evidence in metrics; collectors also verify it against
the sidecar and actual checkpoint. This is evidence of satisfying the stopping
protocol, not mathematical proof of convergence. Subsequent Exp 1/2 hardening
also requires completion evidence at their publication gates.

The Exp 3 Transformer now uses `Linear(128, 2)` on its last timestep, matching
LSTM/S4 prediction-head structure. Its eight attention heads are unchanged.
The `--exp3` flag resolves this explicit model setting in both training and
inference; both sweep launchers supply it. Other experiments keep their default
MLP prediction head. Existing Exp 3 Transformer checkpoints are incompatible
and must be retrained. All sweep runs without completion evidence must also be
rerun (or faithfully resumed to early stopping if compatible).

The linear head removes 8,128 parameters compared with the old 128→64→2 head.
This does not establish any accuracy impact: validation comparisons on the real
data are required. No claim of negligible degradation is made, and no test-set
selection of head type is authorized by this change.

Follow-up verification: 61 tests passed across Exp 3 completion/head integrity,
existing Exp 3 integrity, trainer resume, and launcher failure guards. This
includes real small CPU training loops and model forward/backward checks;
it does not measure real-data forecast accuracy. Diff whitespace checks passed.

Runtime follow-up: `run_exp3.py check` now executes the actual model imports and
a CUDA tensor operation. It exits nonzero when these checks fail, including on
this CPU-only/CUDA-unavailable host. A successful data-only audit is not a
full-launch readiness result.
