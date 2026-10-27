# Intentional behavior differences vs `code/paper included/`

Document any place where the new pipeline deliberately diverges from the
old one. The equivalence test (`tests/test_equivalence.py`) uses this list
to decide whether a numeric mismatch is expected or a real regression.

## Burn-in window

**No longer a difference.** Both pipelines reserve the same history; only the
mechanics differ.

| Pipeline | Behavior |
|---|---|
| `paper included/shared_dataloader.py::add_lags` (old) | Computes lags on the full panel, then drops year 2008 (`df = df[df['Time'].dt.year != 2008]`). With `start=2008-01` that leaves rows from 2009-01 with fully real lags. |
| `src/data/dataloader.py::add_lags` (current) | Computes lags on the full panel, then drops `lag_count` months from `config.start`, at month granularity. With `start=2009-01, lag_count=12` that leaves rows from 2010-01 with fully real lags. |

Both satisfy the same contract:

    first valid target == config.start + lag_count + input_len   (months)

The `input_len` part is **not** deleted. It is the history the first target
consumes, and `TradeDataset` already refuses to emit a target until a full
window fits behind it (`range(len(group) - input_len)`).

### History: this entry previously documented a defect as a correction

Between Phase 0 (`55558a9`, 2026-05-23) and 2026-08-24 this file said the new
pipeline drops `(input_len + lag_count) // 12` **years**, and justified it as
matching the paper. Both halves were wrong, and the entry is kept rather than
deleted because it is why the defect survived three audits.

1. **It applied the window requirement twice** — once by deleting rows, again
   when `TradeDataset` consumed `input_len` months of the survivors. The first
   training target slid from 2013-01 to 2016-01, costing 36 target months on
   every series (~27% of the training set).
2. **Year granularity is wrong** whenever `lag_count` is not a multiple of 12
   or `config.start` is not January. At `lag_count=6` it evaluated to a
   zero-year drop and admitted rows whose lags reach before `config.start`.
   Exp 3 sweeps `lag_count` over {6, 12, 24, 36}, so both paths were live.
3. **The stated justification was false.** The old text claimed the old loader
   "only dropped 2008, so 2009-2011 windows were trained with partially
   zero-padded history." It did not: the drop runs *after* the lagging, so
   `shift(12)` at 2009-01 reaches 2008-01 while 2008 is still present. No
   2009-2011 row was ever zero-padded.
4. **It did not match the paper.** Paper §5.6 states "Years from 2008 to 2011
   were excluded from training as the start of the sequence due to the window
   size of 36 months (which includes the previous 12 lags each)." 2008-2011 is
   48 months = 36 + 12 — the receptive field, consumed as context, first target
   2012-01. Deleting 48 months of rows instead pushes the first target to
   2015-01, which would have required the paper to write "2008 to 2014."

Pinned by `tests/test_burn_in_window.py`. See `debug.md` R33 for the full
verification.

## Lag computation vs. split ordering

**This one is a real difference, and the current pipeline is the correct side.**

| Pipeline | Behavior |
|---|---|
| `paper included/shared_dataloader.py::build_flat_datasets` | Splits first, then calls `add_lags` on each split separately. |
| `src/data/dataloader.py::create_splits` | Lags the full grid once, then forms splits by filtering on target time. |

Lagging each split independently means the leading rows of val and test have
no history inside their own slice, so their lag columns become `fillna(0.0)`.
Val is 12 months, so 2024-01 gets all twelve lags fabricated; test is 5 months,
so every test row's deep lags are fabricated. That is an **evaluation-affecting**
defect, and it is why val/test lag integrity is asserted in
`tests/test_burn_in_window.py::test_val_and_test_rows_do_not_carry_fabricated_lags`.

## Sparse-combo filter

Referenced from `DataConfig.filter_sparse` in `src/data/dataloader.py`; this
section previously did not exist, leaving that citation dangling.

| Pipeline | Behavior |
|---|---|
| `paper included/` | Uses every (State, Commodity, Flow) combination. |
| current | Keeps only combos whose raw `Value` is nonzero in at least `min_nonzero_frac` (default 0.95) of months, measured on the **train slice only** (`cohort_from_train_only=True`). |

Two things to know:

- **The equivalence test must set `filter_sparse=False`** to compare against
  the paper-included population.
- **The measurement window is train-only by design.** Measuring over the full
  timeline is survivorship-biased: on this dataset it drops 15 combos and adds
  zero, and those 15 are exactly the series that thin out during 2024-2025.
  `cohort_from_train_only=False` reproduces the published 903-combo population;
  the corrected default yields 918. See `debug.md` and
  `tests/test_data_pipeline.py::test_cohort_is_frozen_on_train_data`.
- **The vendored S4 loaders use the same window.** `external/s4/**/shared_dataloader.py::build_grid`
  now requires an explicit `cohort_train_end` (its callers pass `split_train_end`) and refuses to
  run the filter without one, so no dataset in that tree can pick a cohort with val/test in view.
  It also excludes the burn-in year that `add_lags` drops, which is what makes it land on the same
  918 lanes as `TradeDataPipeline` rather than 909. Guarded by
  `tests/test_s4_cohort_window.py`. The live arms (`trade_unified.py`, `trade_grid.py`) never used
  this path — they call the canonical pipeline directly.

The 0.95 threshold itself is a **scoping decision that has not been validated**
— it removes 40% of lanes carrying 1.4% of trade value, and the code comment
defending it ("those zeros ... add no signal") is an assertion, not a finding.
See `debug.md` for the open question and the probe that would settle it.

## Impact on the equivalence test

The burn-in entry no longer predicts any drift; if the equivalence test fails
on MAE, look at the sparse filter (population size) and the split-ordering
difference above, not at the burn-in.

## Tabular (tree) arms train on more target months than windowed arms

Not a difference vs `paper included/` — a **within-benchmark asymmetry**,
recorded here because it bears on cross-family fairness claims (2026-09-01
leakage audit; no leakage involved, every extra target is inside train).

- Windowed models (`TradeDataset`) refuse to emit a target until a full
  `input_len`-month window fits behind it: first train target
  `2013-01` → 121,176 train targets over the 918-combo cohort.
- Tabular rows (`src/models/tabular_common.py::extract_from_trade_dataset`)
  need only one prior month (the t−1 feature row, whose 12 lag columns reach
  back further): first train target `2010-02` → 153,306 train rows.

So XGBoost/LightGBM/RF fit on ~26% more supervised targets than the neural
arms, and their effective history per row is 13 months (t−1 row + 12 lags)
against the neural arms' 36-month window. Val and test target sets are
identical across families (11,016 / 4,590), so evaluation is unaffected.
Wherever a tree-vs-neural comparison is framed as capacity- or data-matched,
this asymmetry should be stated.

**Two axes, not one.** Forcing an `input_len` history requirement onto the
tabular extraction is the wrong lever: a tree never consumes a window, so that
would only withhold targets it is capable of predicting. The two asymmetries
are separable and point in OPPOSITE directions:

- *Target volume* (advantages trees): 153,306 vs 121,176. Fixable exactly and
  paradigm-neutrally by `DataConfig.tabular_align_targets`, which clamps the
  tree TRAIN targets to the first month the windowed arms can emit. Features
  are untouched.
- *History reach* (disadvantages trees, heavily): a tree row reaches t-13
  (row t-1 plus its 12 lags); a neural window reaches t-48 (36 rows, each
  carrying 12 lags). Not fixable by target alignment. It moves with
  `lag_count` and must be reported, not silently matched -- reach-matching at
  `lag_count: 47` would trade this confound for a capacity one.

**Measured (2026-09-03, probe P2).** Variant `embeddings`, seeds 26/105/115,
identical val/test (11,016 / 4,590), normalized test metrics:

| model | nMSE native | nMSE aligned | delta | nMAE delta |
|---|---|---|---|---|
| xgboost | 0.051970 | 0.052032 | +0.12% | +0.28% |
| lightgbm | 0.052005 | 0.052181 | +0.34% | +0.30% |
| random_forest | 0.052196 | 0.052379 | +0.35% | +0.36% |

Removing 26% of the tree training set costs 0.1-0.35% nMSE. The direction is
consistent across all three models and both metrics, so the effect is real,
but it is an order of magnitude smaller than the data difference implies and
sits at or below the seed-variation band the sweep already reports (0.5-1.1%).
For xgboost and lightgbm the shift is inside their own across-seed spread.

So the tree arms' data advantage is **not load-bearing** for their standing,
and alignment is cheap: ~0.3% buys a data-matched claim. Caveats: one variant,
one test window, and the seeds vary model RNG only (not data order), so these
spreads are a lower bound on true run-to-run variance.
