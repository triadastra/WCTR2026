# Fair benchmark protocol v1

Status: proposed implementation contract; no new benchmark results yet.

Current Procedia scope decision: the tuning track below is **not adopted**.
Canonical experiments use fixed initial neural LR `1e-4` with the existing
plateau scheduler, no LR search, and separately labelled causal/bidirectional
S4 arms. The authoritative roster is `config/experiment_manifest.env`.

## Questions and comparison tracks

Run all three existing experiments: national aggregate forecasting, disaggregated
panel forecasting, and window-by-lag sensitivity. Preserve the old results and
write this protocol's artifacts to `outputs/fair_v1`.

Report two tracks separately:

1. Controlled architecture: common targets, information, objective, update
   budget and neural optimizer settings. Width/depth matching is not parameter
   matching; measure parameters and memory instead of claiming equal capacity.
2. Equal tuning opportunity: twelve prespecified candidates per trainable arm,
   selected on validation normalized MSE, with the same three tuning seeds
   (26, 105, 115). Report total fitting time, candidate failures and selection
   scores. Never select configurations or seeds using test results. Equal trial
   counts do not imply equal compute; report both rather than conflating them.

The controlled track remains useful even when a model prefers a different LR;
the tuned track answers whether its disadvantage survives equal search effort.

## Data contract

- Fit category vocabularies, cohort membership and normalizers on training only.
- Freeze identical `(series, target month, output)` keys across compared arms.
  Compare hashes and actual key sets before accepting results, not just counts.
- Forecast one month ahead, using observed history available at that date.
  Validation is 2024; final test is January-May 2025. This is rolling one-step
  evaluation, not a five-step forecast made in December 2024.
- Select the panel cohort once from the common training interval, independently
  of window/lag choice. Preserve all-series aggregation for experiment 1.
- For experiments 1/2, use the default 36-month window and 12 lags. For
  experiment 3, use every combination of windows/lags {6,12,24,36}, with
  January 2015 as the common first training target.
- Add a tree window-feature mode: flatten exactly the numeric input window
  supplied to a neural model, then append categorical features once. Keep
  existing last-row tree features as an explicitly separate native-feature
  ablation. The current last-row mode does not match neural history reach.
- Record actual train/validation/test sample counts, valid target counts and
  history reach from each constructed dataset. Do not copy historical counts.

## Model membership

Experiment 1: LSTM, GRU, Transformer, GPT, Mamba-2, local S4, XGBoost,
LightGBM, Random Forest, plus all three deterministic baselines. Categorical
encodings become constant on the aggregate series; run one canonical encoding
instead of treating duplicate encodings as independent evidence. ND architectures
have no nontrivial categorical axes here; mark them structurally inapplicable
rather than inventing a grid or silently omitting them.

Experiment 2: all existing flat, CA, FiLM, axial and ND arms, plus tree
window-feature arms and the three deterministic baselines. Distinguish
Transformer's flat CA/FiLM implementation from GRU/LSTM panel attention.

Experiment 3: extend the existing three-backbone sweep to all applicable
experiment-2 trainable arms, using the same 16 cells and three seeds. Include
baselines as reference predictions; do not claim identical baseline predictions
are independent trials. Report native-tree ablations separately.

## Optimization and architecture controls

- Neural controlled defaults: AdamW, LR 1e-4, weight decay 1e-4, author-marked
  SSM exemptions, FP32, norm clipping 1, plateau patience 15/factor 0.5,
  early-stopping patience 50/min_delta 1e-4, maximum 5000 epochs.
- Validate once per epoch. Record actual optimizer steps and targets per step.
  Include incomplete batches rather than silently dropping training targets.
- Panel batches should contain one complete target month, including grouping
  all flows or commodity-flow slices before a step. A shuffled flat batch of
  918 targets matches count but not the target-month composition of a panel
  batch. Use month-aware sampling or explicit accumulation with valid-target
  weighted loss; do not average unequal masked microbatch means.
- Aggregate models share one batch policy: batch 33, four equal steps over all
  132 windows (`config/exp1.yaml`; the published 64 left a partial batch of 4
  that the loader dropped every epoch), and the aggregate path never drops a
  partial batch (`data.train_drop_last: false`). This doubles updates per epoch
  relative to the previous batch 66, including the update budget within the
  unchanged epoch-based scheduler and stopping patience. Validation has not
  established which batch size performs better.
- Positional encoding: sinusoidal for the Transformer and GPT families. The
  shared `PositionalEncoding` applies dropout equal to the model dropout (0.1)
  after adding positions (the Vaswani et al. recipe); GPT's causal encoder
  applies none. This is a Transformer-family control, active since 2026-09-08
  and recorded per run as `positional_dropout` in the run log.
- Align local S4 and external S4 state size: 64 for the whole S4 family
  (settled 2026-09-13; the S4/S4D default, and the 36-month time window needs
  at most 36 modes per kernel). One known consequence: the S4ND Commodity-axis
  kernel in the 3d/4d arms spans 93 positions, so at N=64 that axis's filter
  is not fully expressive; report it as a property of the setting.
  Keep Mamba's native state mechanism separately described; its state dimension
  is not numerically comparable to a DPLR state dimension.
- Resolve nested CA/FiLM settings into the parameters actually consumed by
  constructors. Save the resolved architecture and measured parameter count.
- Keep fixed-depth Mamba-ND as primary; report achieved axis/direction coverage.
  Full-direction coverage at greater depth is a separately labelled ablation.
- Causality. The ND arms (S4ND with `bidirectional: true`; Mamba-ND where its
  schedule achieves both directions) are two-sided over the input window; the
  target is outside the window, so this is not leakage, but every 1-D
  comparator except the unmasked vanilla Transformer and the separately
  labelled `s4_bidir` controls is causal, and at
  fixed depth 4 Mamba-ND achieves two directions only for Time+State at 2d,
  Time at 3d and no axis at 4d. Any "ND vs 1-D" statement must say so. Each
  run records `architecture.bidirectional` (achieved, not requested) and, for
  Mamba-ND, `architecture.scan_coverage` in `metrics.json`; the results table
  carries a `bidirectional` column. For a two-sided-vs-two-sided comparison
  within the S4 family, the canonical manifest now pairs every causal S4 arm
  with `s4_bidir`, including embeddings, one-hot and axial variants.
- Tree candidate budgets refer to boosting rounds/trees, not neural epochs.
  Enable LightGBM row bagging explicitly when subsample is below one.

## Tuning and reporting

Prespecify search spaces in a versioned machine-readable manifest before runs.
Neural candidates should vary LR, weight decay and dropout; tree candidates
should vary learning rate (boosters), depth/leaves and feature/row sampling.
Use the same candidate count and seeds per applicable arm. Keep architecture
dimensions fixed during this first search to avoid hidden capacity tuning.
Tune separately for experiments 1 and 2. Freeze experiment-2 selections for
every experiment-3 window/lag cell; retuning per cell would confound the
sensitivity measurement and multiply the search budget.

Save candidate validation results, selected configuration, resolved config,
source/data hashes, checkpoint provenance, runtime, peak memory, parameter
count and failures. Evaluate the selected candidate's three seed checkpoints
on test only after selection is frozen. Report seed mean/std and prediction
ensemble separately. Report normalized MSE as primary, alongside original-unit
MAE/RMSE and seasonal MASE. Significance: the unit is the lane (state x
commodity x flow, one mean loss over its five test months) and the state is
the cluster -- cluster-robust paired t with G-1 degrees of freedom and a wild
cluster bootstrap over the 8 states (`ext/eval/resampling.py`), BH/Holm
across the arm family. Do not collapse lanes to state means (8 blocks leave
every test powerless) and do not pool the 4,590 points. With 8 clusters the
exact wild bootstrap's smallest p is 2/256 = 0.0078, so Holm (family-wise)
across ~45 arms is floored above 0.05 by construction; report BH q-values as
the screening verdict and Holm only as the stricter reading. Friedman/Nemenyi
over lanes is a descriptive rank grouping. Use paired target-month comparisons; five test
months limit confidence and must not be treated as thousands of independent
time observations. Do not declare a winner from seed variance alone.

Completion requires actual successful runs for every applicable declared cell,
coverage and provenance validation, and comparison tables for all experiments.
A manifest, smoke run or historical workbook is not a completed benchmark.

## Current execution prerequisites

The inspected host has an RTX 5090 Laptop GPU with about 24 GB VRAM. Bundled
Python has NumPy/Pandas/YAML but no Torch, XGBoost or LightGBM. WSL lists only
Docker Desktop (stopped). A Linux CUDA training environment must be established
and all backend smoke checks must pass before scheduling the complete suite.
