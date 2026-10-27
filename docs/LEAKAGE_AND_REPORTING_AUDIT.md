# Leakage and automatic reporting audit — 23 September 2026

The defensible claim is **a retrospective benchmark with train-only
preprocessing and target-excluding historical inputs**. An unconditional
"leakage-free real-time forecast" or "untouched final holdout" claim is not
supported by this source. Changing the reporting label does not resolve those
experimental limitations.

## Leakage checks

The canonical native and external S4/S4ND loaders select the cohort, fit
normalization, and build category mappings on training data. Features for
target month t exclude t and later months. Earlier test outcomes may enter
later input windows under the stated rolling one-step protocol; the fitted
model is not retrained on those test targets. Bidirectional processing within
the historical input window does not expose the future target.

Regression checks perturb held-out values and compare the actual training,
validation and test inputs, not just calendar labels. Coverage includes native
flat/panel windows, aggregate models, both tree encodings and full-window tree
features, plus all six external adapters (S4 embeddings/onehot and four S4ND
layouts). The external test changes March 2025 and later outcomes: earlier
train/validation data and test inputs through March must be identical, while
March targets must change. This positive control makes an ineffective mutation
fail the check. Loss tests independently check valid-cell masking and gradients.

Two limitations cannot be removed from the existing five-month test set:

1. It was inspected while code and design were being changed. It is therefore
   retrospective evidence, not a newly untouched confirmatory test. Fixing this
   requires freezing the protocol and evaluating an uninspected later period.
2. The pinned monthly snapshot does not contain per-observation publication
   times and revision vintages. Census publishes statistical months later and
   revises data. A window ending at t−1 does not prove that its values were
   available at the claimed forecast origin. A real-time study needs archived
   vintages and availability-date filtering, including preprocessing fitted
   from data available at that origin. Adding a guessed delay would not solve
   revision leakage.

Primary source documentation: [Census release schedule](https://www.census.gov/foreign-trade/schedule.html)
and [revision policy](https://www.census.gov/foreign-trade/guide/revisions.html).
These describe publication timing and revisions; they do not authenticate the
vintage of each record in this repository. Mixed import/export measures and
historically summed ambiguous duplicates also remain source-definition limits.

## Reporting corrections

- Seed means and sample standard deviations use full stored precision.
  Excel formatting controls display only. Small nonzero numbers use scientific
  notation instead of appearing as rounded zero.
- Ensembles require the entire validated canonical seed roster and identical
  keyed truth. Missing files, duplicate keys and mismatched populations fail.
  All direct and seasonal MASE/RMSSE metrics are recomputed after averaging
  predictions; averaging seed losses is a different statistic and remains in
  the seed-mean table.
- Model parameter counts must agree across seeds for the same arm; counts
  are metadata, never averaged into a fractional model size.
- The per-seed CSV checks directory/model/variant/seed identity and training
  completion as well as metric/source validity. It publishes atomically and
  rejects an empty collection.
- The two automatic cross-model graphs use the same validated rows and
  arithmetic as the workbook. They do not read a stale cached CSV or copy
  unchecked old PNGs. Similar observed loss is not labelled statistical
  equivalence. Existing historical plot folders are not recertified.
- Every written workbook cell is read back and compared before publication
  (within Excel's approximately 15-significant-digit numeric precision).
  Input hashes are rechecked before replacing the prior workbook. A failure
  leaves the prior workbook intact; a prior workbook can still be historical.
- Missing GPU checks, completed runs or source evidence are not converted
  into passing results. `--claim real-time` and `--claim confirmatory` fail
  before any workbook is written for this benchmark.

## What the generated workbook contains

`INDEX` explains each sheet. Headline seed means, every seed's metrics,
ensemble metrics, and the complete window/lag grid remain separate.

`Run_Details` provides one row per run: model, variant, seed, recorded parameter
counts, width/layers/state size/dropout where applicable, optimizer and initial
learning rate, weight decay, configured batch size, input window/lags, split
dates, validation-selected epoch/loss, stop reason and checkpoint/source hashes.
Configured batch size is not relabelled as targets per optimizer step.

`Run_Values` exposes all stored evaluation and resolved-contract fields, with
typed values and source paths. It includes auxiliary saved metadata and
per-field epoch-log summaries, explicitly labelled as recorded or derived.
The original epoch logs retain every individual observation; summaries are
not test metrics. They may span multiple attempts. Missing values are labelled
"not recorded" rather than invented. Metadata availability does not itself
prove independent measurement accuracy or runtime equivalence.

New inference runs also record total/trainable/frozen parameters, parameter
and buffer bytes, and each named tensor's shape/dtype/trainability in
`Model_Tensors`. Native tree estimators report their own parameters and
available tree/node/leaf or boosting-round counts; they are not described as
zero-parameter neural models. Weight values stay in the original checkpoint,
not thousands of spreadsheet columns. Older artifacts without these fields
remain explicitly missing rather than receiving guessed numbers. The legacy
`num_params` field counts trainable native parameters and total external S4
parameters; the new explicit total/trainable columns remove that ambiguity.

`Report_Files` records paths and SHA-256 hashes of report inputs.
`Metric_Definitions` distinguishes normalized scale-dependent errors, physical
percentage errors and train-scaled seasonal metrics. In particular, combined
RMSE is the mean of channel RMSEs, not the square root of combined MSE.
`Evaluation_Scope` carries the retrospective/availability limitations into
every generated workbook. Large metadata tables split into numbered sheets
instead of truncating rows at Excel's limit.

## Rebuild without training

After all current production artifacts pass their contracts:

```bash
python scripts/build_paper_results.py --repo . --out results.xlsx
python scripts/build_graphs.py
```

The normal `scripts/run_all_experiments.sh` controller already runs collection,
workbook generation and graph generation after all experiments and artifact
checks complete. A graph failure is now an explicit reporting failure instead
of a successful overall completion message. Do not bypass the canonical roster
to produce a partial paper workbook. Short diagnostics and historical outputs
are intentionally insufficient.
