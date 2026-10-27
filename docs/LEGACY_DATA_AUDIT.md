# Original dataset audit — 2026-09-05

## Preservation and scope

User decision: retain the original dataset; stop replacement-data downloads.
No download process was running at this audit. No original CSV was edited,
deduplicated, replaced, or regenerated. Previously downloaded staging files
were not installed or deleted.

All 16 CSVs under `data/imports` and `data/exports` were compared byte-for-byte
with Git commit `c5ce1b9db40a50989024ef77ed7b138037f21354`: all match.
This establishes repository preservation, not independent identification of
the exact snapshot used at the conference.

The files contain 1,005,427 records across eight states and 97 HS2 chapters
in aggregate. Every file spans 209 months, January 2008–May 2025; individual
files need not contain all chapters. The experiment period remains January
2009–May 2025, with training through December 2023, validation in 2024,
and testing January–May 2025.

The historical loader selects total-vessel imports and containerized-vessel
exports. It treats blank numeric cells as zero and sums regions into monthly
state–commodity–flow observations. Neither that blank interpretation nor the
source's factual accuracy is independently certified by this audit.

## Duplicate findings

A source key is flow, state, HS2, country/region, and month. Different regions
for the same state–commodity–month are not duplicate source keys.
Checks using both full commodity labels and extracted HS2 codes agree.

| Files | Records | Excess records with repeated source keys | Exact full-row duplicate excess |
|---|---:|---:|---:|
| All eight import files | 443,184 | 0 | 0 |
| Arkansas exports | 45,083 | 2,250 | 2 |
| Other seven export files | 517,160 | 0 | 0 |

Arkansas exports contain 2,250 duplicate pairs (4,500 involved records).
After historical blank-as-zero numeric conversion, 2,198 pairs disagree in
at least one selected containerized target. The other 52 pairs have equal
selected targets; only two pairs are identical across every original column.
Full-row duplicate CSV record numbers, counting the header as record 1,
are 26900, 27121, 29192, and 29402.

Duplicates occur from 2008 through 2025, including validation and test periods.
No missing source-key fields, malformed nonblank selected numeric values,
or negative selected targets were found. Numeric blanks do exist.

## Sensitivity diagnostic, not a proposed correction

For Arkansas exports during January 2009–May 2025:

| Selected target | Historical sum | Arbitrarily keep first per key | Arbitrarily keep last per key |
|---|---:|---:|---:|
| Containerized value (USD) | 28,479,676,325 | 27,240,511,867 | 27,126,822,613 |
| Containerized weight (kg) | 12,212,709,969 | 11,762,725,630 | 11,862,196,548 |

Those arbitrary choices reduce Arkansas export value by 4.35–4.75% and weight
by 2.87–3.68%. They are not evidence that either record is the correct one,
and no such change was applied. These are source-panel sums before cohort
filtering; model-level effects remain unmeasured.

## Readiness

The original source files are preserved and available for historical
reproduction. They must not be described as duplicate-free or independently
validated ground truth. Conflicting Arkansas records require source-lineage
review before any corrective deduplication.

## Explicit reproduction entry point

`config/legacy_reproduction.yaml` is an opt-in overlay: original mixed targets,
97 chapters, unchanged dates, full-timeline 903-series cohort, and `legacy_sum`.
The latter requires a source fingerprint, warns on duplicates, and saves the
duplicate counts in source provenance/run contracts. Strict mode remains
`duplicate_policy: error`. No row is removed under the legacy option.

Run a read-only preflight using a Python environment with the project dependencies:

```bash
bash scripts/run_legacy_reproduction.sh check
```

The separate launcher does not call the Census downloader or require a new
download receipt/ground-truth artifact. It routes `train` and `infer` to the
root Python entry points with `--legacy-reproduction`. For example:

```bash
bash scripts/run_legacy_reproduction.sh train --model lstm --variant embeddings \
  --seed 26 --out-dir outputs/legacy_reproduction/lstm_embeddings_seed26
```

New reproduction outputs must be run subdirectories of
`outputs/legacy_reproduction`; published output directories are rejected.
Use the same flag for inference. The profile preserves the aggregate batch
setting when `--aggregate` is selected; the 903 per-series batch setting does
not apply to a single aggregate series.

This is historical **data** reproduction, not a claim of identical historical
optimization, checkpoints, model rankings, or conference results. Other
training fixes remain in the working tree. The main Exp1–3 sweep controllers
now also use the original pinned mixed measures and audited sum-all policy;
their replacement-download receipt/panel gates have been removed. They retain
the corrected training protocol, including the train-only cohort for per-series
experiments, so they are not exact conference-reproduction launchers. Use the
separate legacy entry point when requesting the historical 903-series cohort.
No training sweep was started by this change.

Verification: seven focused regression tests passed. The selected raw values,
monthly grid, and filtered panel compare exactly with the committed loader;
the retained cohort contains 903 series. The aggregate dataloaders emit 132
training, 12 validation, and five test samples with finite targets. Source
hash mismatch and unpinned legacy mode are rejected. These tests do not
establish full training equivalence or resolve the conflicting source rows.

## Source-bound Exp1 result validation

The canonical Exp1 skip gate and paper collector now reconstruct the five
aggregate test targets from the pinned CSV bytes. They check normalized targets,
inverse-scaled targets, prediction inverse transforms, and all MASE/RMSSE
components against training-only seasonal scales. A self-consistent prediction
file with invented targets is therefore insufficient for publication or skipping.
The collector reconstructs once per invocation, not once per seed.

This reference reproduces the historical sum-all dataset; it does not resolve
the ambiguous Arkansas records or certify Census ground truth. Neither the
reference reconstruction nor validation writes source files or downloads data.

## Training/runtime audit boundary

CPU end-to-end checks exercise training, checkpoint reload, and source-bound
Exp1 evaluation for LSTM, GRU, Transformer, GPT, S4, XGBoost, and LightGBM
(embedding and one-hot variants), plus all three classical baselines. Learned-model checks
are smoke runs, not completed paper runs; the canonical paper gate rejects
their shortened recipes. Mamba CUDA training is not covered by those CPU
end-to-end checks.

S4's standalone logger no longer imports the Lightning training stack. Mamba's
temporary kernel import overrides are restored on success and failure, and
new partial Mamba imports are discarded on failure. This tests import hygiene,
not GPU numerical correctness: this machine lacks CUDA/Triton.

Rejected neural resumes now leave existing log/status files and source sidecars
unchanged. Logger initialization follows checkpoint validation/restoration;
restored random streams are preserved around tracker initialization. Native
tree training reports measured training MSE, not a placeholder zero.

Verification on 2026-09-05: 336 regression tests passed after the runtime and
resume fixes, excluding the two end-to-end test files. Those files' seven
learned-model smoke cases, three baseline smoke cases, and one configuration
case passed in the preceding suite run. That run stopped later on three
Mamba fixture failures; fixtures now stub the explicit runtime-loading boundary,
and the corrected tests are included in the clean 336-test rerun. A separate
post-resume-fix LSTM train/reload/evaluation check also passed. Shell syntax and
`git diff --check` passed; raw CSVs still match the pinned reference commit.

## Follow-up Exp1 protocol checks

All seven one-hot learned-model round trips subsequently passed. A synthetic
future-target perturbation test checks both tree encodings and the neural
window path: changing October's target leaves training, validation, and
October's inputs exactly unchanged, but changes November's observed history.
This documents rolling one-step evaluation, not a fixed-origin five-step
forecast or a real-time publication-delay backtest.

Neural training now rejects unavailable requested CUDA/MPS devices rather than
silently writing a CUDA-labelled contract for a CPU run. Explicit CPU runs and
native CPU tree estimators remain supported. Aggregate contracts now record
the same disabled-TF32 arithmetic policy as the actual runtime; older contracts
without that policy cannot pass the canonical gate.

Exp1 inference now records checkpoint-bound completion evidence. Neural paper
gates require a finished early-stopped run, while completed epoch-limit smoke
runs remain inspectable but not publishable. Native tree completion checks
both estimator file hashes and their saved run contract, without deserializing
joblib in the paper gate. Baselines do not need a training checkpoint. Both the
skip gate and paper collector enforce these checks; the receipt supplements,
not replaces, source-bound target validation.

Follow-up verification: 347 regression tests passed (the two end-to-end files
excluded), and the newly added collector enforcement test passed separately.
All seven one-hot CPU round trips passed; fresh LSTM and XGBoost one-hot round
trips passed again after completion enforcement was added. Both real Mamba
encoding cases are now present in the CLI test matrix and explicitly skipped
here because CUDA is unavailable. On a configured CUDA/Triton machine, run
`python -m pytest tests/test_exp1_neural_end_to_end.py -k mamba -q` to exercise
them; these are one-epoch checks, not full paper training.

Compatibility: older Exp1 artifacts lacking the explicit arithmetic policy or
checkpoint-bound completion evidence are not accepted as current paper runs.
No existing output artifact or original source CSV was rewritten by the audit.

## Resume finalization follow-up

A regression test reproduced failure when a non-cosine run increased its epoch
ceiling but never improved on the earlier best checkpoint. The selected weights
kept the old ceiling in their embedded contract. Finalization now permits only
that compatible ceiling change, preserves the selected tensors/epoch/score,
and records `selection_origin_run_contract` when rebinding the checkpoint to
the extended run. Different data/configuration and reduced epoch ceilings are
not accepted by this path. This metadata update applies to a user-requested
resume, not to the original source data or a blanket artifact migration.

The run emits its finished message/status only after completion publication
succeeds. A failed receipt write no longer leaves a falsely successful terminal
status. Completion schema checks reject boolean version numbers, malformed
epoch/patience fields, invalid SHA-256 strings, and non-object native metadata.

Verification: the no-new-best extension test failed before the fix and passed
afterward with exact tensor equality. The focused training/completion group
passed 67 tests; the broader regression suite then passed 363 tests (excluding
the two end-to-end files). A fresh LSTM embedding CLI train/reload/source-bound
evaluation passed separately. Raw CSV comparison, shell syntax, and whitespace
checks remained clean. Mamba CUDA verification remains outstanding.
