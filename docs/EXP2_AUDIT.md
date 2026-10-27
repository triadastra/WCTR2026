# Experiment 2 audit and launch guide — 5 September 2026

Experiment 2 needs a corrected rerun. Its historical rankings are not validated
by this audit. The pipeline is prepared and checked on the pinned original
source snapshot; this Mac cannot run the full CUDA/Mamba benchmark.

## Findings and changes

22 September launch clarification: the current manifest has **57 arms × 3
seeds = 171 runs**, including separately labelled causal and bidirectional S4.
The active input is the monthly release with fingerprint
`0e8c268887a4d98c484c4a053326347cae75f74431bec0133577d37ff77ee6fb` and
`duplicate_policy=error`; the historical raw-source audit below describes the
original reports whose summed values were preserved in that release.

16 September follow-up: [S4ND occupancy and dimensionality contract](S4ND_PADDING.md)
defines masking inside every grid layer, removes the redundant loss multiplier,
and invalidates checkpoints trained with the old padding/loss behavior. The
historical measurements below retain their original audit context.

| Finding | Consequence | Resolution |
|---|---|---|
| Historical full-timeline sparse-series selection | Validation/test availability influenced inclusion: 903 historical series versus 918 selected using training only. | Existing train-only selection retained; future-value perturbation tests verify training data and cohort invariance. |
| S4ND zeroed padded predictions but reduced MSE over the entire grid | Padding diluted gradients and the absolute stopping threshold; real-cell fraction differed by representation. | Select valid prediction/target pairs before loss/metrics; weight epoch loss by valid target count. All four grid layouts tested. |
| Inherited S4 kernel LR was 0.001 despite global LR 0.0001 | S4 parameters used a private optimization recipe. | Disable the kernel LR override; derive shared LR, schedule, stopping and clipping settings from base configuration. |
| Upstream S4 exempted biases/norms through generic optimizer hooks | Extra decay exemptions differed from the root trainer. | Disable generic hooks for this benchmark; preserve explicit SSM parameter exemptions. |
| S4 used state size 128, while root S4/ACA used 64 | “Identical backbone” was overstated. | Launcher sets S4/S4ND state size to the root S4 configuration (64). Other architectural differences remain disclosed below. |
| S4 explicitly enabled TF32 | fp32 labels did not establish equal arithmetic settings. | Explicitly disable TF32 in both runtimes. |
| S4 evaluator rebuilt current defaults, tolerated parameter mismatches and emitted obsolete v2 metrics | Weights could be evaluated under different settings; percentage metrics and required fields differed across families. | Save an immutable training configuration and full contract inside checkpoints; enforce it on resume/evaluation. Use shared normalized losses, real-unit percentages, and training-history MASE/RMSSE. Missing parameters fail. |
| S4 failure handling and checkpoint-complete markers were insufficient | Failed/missing evaluation could return success; old checkpoints could be skipped. | Nonzero failures, seed-specific evaluation, stale-marker removal, exact recipe checks, and checked resume. Each arm must finish evaluation before success. |
| Exp 2 collection did not enforce the complete paired roster or full recipe | Partial seeds, smoke recipes or differently trained artifacts could be averaged. | Paper collection requires 57 arms × seeds 26/105/115, exact contracts, verified prediction metrics, and matching test-target fingerprints. |

Current training/validation/test windows are chronological. Features for target
month t end at t−1; lag creation uses backward shifts. Train-only normalization
and category maps remain in place. Tests alter future outcomes and verify they
cannot alter training inputs/statistics or the input window used to predict t.
S4 validation already excludes the test loader. Final last-epoch test execution
is now disabled by the benchmark launcher; evaluation uses the validation-selected
checkpoint. Earlier observed test results cannot be “unseen”: further model
choices must be described as retrospective, or confirmed on a new holdout.

## Measured data and optimization schedule

The historical raw-source preparation check verified:

- SHA-256 source fingerprint: `e5c6610ff4d751fa18e313b06fed04b6c3846ee0d1776417b2b68af4f99a264d`.
- 918 kept state × commodity × flow series, selected through December 2023.
- 36 history rows and 12 lag features; training targets January 2013–December 2023.
- 121,176 training targets; 11,016 validation targets in 2024; 4,590 test targets in January–May 2025.
- Flat and combo loaders both use 132 updates per epoch. Flat batch size is 918; a combo batch contains one complete month.

The source is explicitly `legacy_mixed` with `legacy_sum`: the pinned original
CSV records and original measure semantics are preserved. Its audit reports
2,250 ambiguous duplicate report keys (4,500 rows). Imports and exports use
different vessel-measure scopes in this historical dataset. These are source
limitations, not corrected ground truth. All models receive the same audited
source; do not describe it as independently validated total-vessel data.

## Limits on fairness claims

This is a **shared-data, fixed-recipe model comparison**, not a matched-parameter
or matched-compute causal experiment. Equal width, nominal depth and updates do
not equalize parameters, FLOPs, memory or training time. S4ND and ACA still differ
in their grid operations, temporal treatment and internal layer structure.
Report actual parameter counts, already exported for neural/S4 results.

Trees receive the same supervised target dates and the full historical window,
flattened into tabular features (`tabular_feature_mode=window`). Their categorical
IDs are ordinal or one-hot, not learned neural embeddings, and their inductive
biases still differ from sequence models.
A common learning rate is a reproducible fixed recipe, not evidence that every
model received equally effective tuning. Boosting rounds and neural epochs are
not equivalent compute budgets.

Every neural representation uses one complete target month per update: 918
valid targets, with no cross-month sample mixing. Month ordering and grid
coupling can differ; the internal operations are not identical.
Mamba-ND's four-layer schedule also gives different directional coverage as rank
increases; the existing `depth_policy=fixed` must remain disclosed.

The five test months are not 4,590 independent temporal observations. Seeds
measure optimization variability, not uncertainty over future economic periods.
This benchmark assumes prior monthly observations are available at forecast time;
no release-vintage or reporting-delay audit was performed.

## Run preparation and training

From the repository root, inspect the roster without starting training:

```bash
python scripts/prepare_exp2.py
```

On the CUDA training host, check the source, model recipes, and dependencies:

```bash
python scripts/prepare_exp2.py --data-dir . --check-data --check-runtime \
  --report outputs/exp2_readiness.json
```

Under the current monthly-release protocol, an external data root must contain
`data/processed/hs2_monthly_aggregated.csv`. Use that same root with `DATA_DIR`
during training. The check exits nonzero on failure.
A plan-only pass is not reported as ready to train.

Run all paired seeds, stopping if a seed fails:

```bash
for seed in 26 105 115; do
  DATA_DIR=. RESUME=1 bash scripts/run_full_test.sh "$seed" || break
done
```

The launcher trains/evaluates all 57 arms per seed, uses a per-seed controller
lock, and checks each result before reuse. A stale S4 checkpoint is retrained;
a valid interrupted checkpoint can resume. If a controller was forcibly killed,
verify it is no longer running before removing its empty `.seed<N>.lock` directory.

After all seeds finish:

```bash
python scripts/prepare_exp2.py --check-results
python scripts/collect_results.py --runs-dir outputs/full
```

The overall paper builder additionally requires completed Experiments 1 and 3.
Keep exploratory runs outside `outputs/full`; the final roster gate rejects extras.

## Validation performed

### Launch hardening follow-up

Exp 2 now verifies terminal training evidence during inference, result reuse,
readiness checking and paper collection. Root neural models use the shared
checkpoint-bound completion receipt; native trees use their atomic bundle
metadata and hashes. Upstream S4/S4ND writes its receipt only after Lightning
`fit` returns, binding the selected validation checkpoint, run contract, epoch
count and stopping state. Epoch-capped or interrupted runs cannot pass the
paper gate.

The completion writer also compares the terminal best validation score against
the selected checkpoint's saved ModelCheckpoint callback state. Missing,
nonfinite or mismatched selection scores cannot create a new completion record.
Nine focused completion tests cover this check and the earlier contract,
checkpoint-tampering and stopping-policy checks.

Runtime preflight executes imports in isolated interpreters and constructs a
Lightning trainer, rather than only checking whether packages can be found.
The master runs this check before starting Exp 1. The local Lightning 1.9
installation rejects the configured `strategy: auto`; use the upstream pinned
2.0.4 environment and rerun preflight. Missing `torchmetrics`, `platformdirs`
and `psutil` were also exposed locally. When `wandb=null`, the trainer now
explicitly disables logging instead of unexpectedly enabling TensorBoard.

Full-launch readiness remains conditional on actual CUDA/Triton execution,
Mamba-family kernels and production-shape GPU memory checks on the training
host. No full experiment sweep has been launched by this audit.

Follow-up verification: the broad regression run passed 374 tests (excluding
the separately exercised Exp 1 neural/baseline end-to-end files); the updated
runtime/launcher group passed 14 tests. A temporary environment with Lightning
2.0.4 and the missing dependencies passed all non-GPU runtime probes. Its only
remaining failures were Triton/Mamba imports and CUDA execution. This does not
repair or certify the training host's installed environment.

- Experiment 2 regression tests cover future-value isolation, all S4 grid-loss
  masks and gradients, valid-count epoch weighting, canonical seed/recipe checks,
  missing-seed rejection and evaluator failure status.
- Related metric, optimizer, tree-target-alignment, checkpoint, launcher and
  provenance tests passed (41 tests in the final shared regression group;
  24 in the final Experiment 2/launcher group, with overlap).
- All 918 real-source series have nonzero training seasonal scales for the
  MASE/RMSSE metrics.
- All six upstream S4/S4ND architectures passed a synthetic CPU Lightning 2.0.4
  training fit, early stop, checkpoint save/reload, completion-receipt validation
  and evaluation cycle. The smoke fixture uses zero learning rate/dropout to
  make stopping deterministic; this tests lifecycle wiring, not convergence or
  accuracy. Reproduce with
  `cd external/s4 && python ../../tests/smoke_exp2_s4.py` in the upstream runtime.
- Real-data preparation passed with the counts above. Full production training
  was not launched: this host lacks a CUDA GPU and Triton. CUDA kernel execution,
  production memory requirements, convergence and corrected rankings remain to
  be established on the training host.
