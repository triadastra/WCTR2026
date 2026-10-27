# Combined review integration — 2026-09-13

The reviewed changes are integrated on `codex/fair-comparison-v4`, based on main
6f55858. The earlier independent snapshot is retained at
`codex/review-snapshot-20260913` (2d6dece). Review: PR #21.

## Settled protocol

The primary objective and paired comparison use lane-weighted MSE over valid
series-time observations, with equal value/weight channel weights. States affect
the uncertainty calculation, not the primary mean. Equal-state differences and
Friedman/Nemenyi ranks are labeled sensitivity analyses. S4 state size is 64
throughout the family. Exp 2 contains 52 arms and three paired seeds (156 runs).

## Integrated repairs

- Exact source, calendar, cohort, recipe and checkpoint-completion checks precede
  artifact reuse or publication; missing seed populations fail closed.
- Checkpoint selection/resume, tree serialization, vendor imports, batch/mask
  handling and architecture metadata are reconciled with current main.
- Source preflight accepts the preserved canonical source; full-timeline legacy
  cohort reproduction requires explicit opt-in.
- Exp 3 requires the complete sweep and computes observations from its results.
- Statistical figures and paper figures read validated artifacts. Historical
  fixed winners and S4 parameter counts are removed. Cached workbook mutation
  commands and the old five-month pseudo-DM report cannot publish new output.

## Validation

The integrated full CPU run completed with **581 passed, 2 failed, 2 skipped**
and 55 warnings in 1,011.61 seconds. The two failures were addressed in a
**15-test follow-up that passed** (80.73 seconds):

- The strict duplicate-source fixture retained a contradictory legacy opt-in.
  It now selects the strict train-only profile; added opt-in boundary tests pass.
- LightGBM onehot inference was terminated with SIGTERM and no Python exception.
  Its unchanged original-data training/inference/validation round trip passed
  when rerun. No model code was changed to hide that termination.

The full suite was not repeated after the fixture-only correction. The two
skipped cases require the unavailable Mamba/CUDA/Triton runtime. Local logs:
`/tmp/wctr-repair-full-20260913.log`,
`/tmp/wctr-repair-final-rerun-20260913.log`,
`/tmp/wctr-report-source-binding.log`, and `/tmp/wctr-integration-render.log`.

Additional validation: 61 reporting/audit/weighting checks and 27 statistical
consumer/rendering checks passed. Undefined-name lint, shell syntax and Git diff
checks passed. All 16 original CSV files match their pinned byte lengths and
SHA-256 hashes. Input CSVs and saved historical experiment results are unchanged.

## Evidence limits

This integration supplies software repairs, not corrected benchmark rankings.
CUDA/Triton/Mamba GPU qualification and canonical reruns remain outstanding.
The preserved target mixes import/export measures and sums duplicate reports;
source ambiguity is documented in SOURCE_FIELD_AUDIT.md. The already-inspected
holdout cannot become an untouched confirmatory test through a code change.
Main is unchanged until the review branch is merged.
