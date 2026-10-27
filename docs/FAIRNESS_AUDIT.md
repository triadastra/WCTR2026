> Superseded for Experiment 2 by [the 2026-09-05 audit](EXP2_AUDIT.md).
> In particular, grid-loss reduction and S4 optimizer equality were not established by the checks below.

The 903-series / 144-window counts below describe the older protocol, not
the current canonical defaults. Current per-series runs use the train-only
918-series cohort and 132 training target months. Exp1 uses one aggregate
series with 132 training, 12 validation, and five test targets. See
`config/base.yaml`, `config/exp1.yaml`, and `docs/LEGACY_DATA_AUDIT.md`.

# Fairness audit — models and training protocol (2026-07-27)

> **Numbers below are PRE-CORRECTION.** This audit was written on the 903-combo
> full-timeline cohort: it quotes 903 lanes, 130,032 train targets and 144
> steps/epoch throughout. The corrected train-only cohort is 918 lanes, 121,176
> train targets and **132** steps/epoch. The *arguments* are unaffected — every
> one is about a ratio between arms — but do not quote a figure from this file.
> See `RESULTS_STATUS.md` and the 2026-09-03 audit.

A pass over every model and every stage of the pipeline, looking for anything
that advantages or handicaps a model for reasons unrelated to the architecture
being compared. Written before the post-correction re-run so the decisions can
be made once.

Severity: **S1** = points at a headline conclusion; **S2** = affects a specific
comparison; **S3** = worth documenting.

---

## S1-A. Combo arms train at a 56x larger effective batch with an unscaled learning rate

**First, what this is NOT.** Both formulations see *exactly the same data* each
epoch: 903 combos x 144 windows = 130,032 prediction targets for the flat
loader, and 144 months x 903 combos = 130,032 for the combo loader. Identical.
The multidimensional arms are not shown less data.

What differs is how that data is cut into batches:

| | batch | steps/epoch | targets per gradient step |
|---|---|---|---|
| flat | 64 | 2,032 | 64 |
| combo | 4 | 36 | **3,612** |

So the combo arms train at a **56x larger effective batch** — at the same
`lr=1e-3`, unscaled. That is the textbook large-batch problem: raising the
batch without raising the learning rate under-trains for a given epoch budget.
`patience=10` compounds it, since it is counted in epochs: the combo arms stop
after ~360 steps without improvement, the flat arms after ~20,320.

**This cannot be fixed with batch size.** One combo sample structurally
contains all 903 combos' targets, so even `batch=1` leaves an effective batch of
903 — still 14x the flat arms'. The panel formulation forces it. That is the
real content of "higher dimensionality just has fewer samples": true, and
unavoidable, which is precisely why the learning rate has to absorb it.

**The lever is the learning rate** (or a longer schedule), and scaling LR with
effective batch is standard, citable practice rather than per-model tuning — so
it is defensible in the paper in a way that hand-tuning one model would not be.

The original sampling detail, for reference:

| dataset | one sample is | train samples | batch | **steps/epoch** |
|---|---|---|---|---|
| per-series (flat: `embeddings`, `onehot`) | one (combo, window) pair | 903 x 144 = 130,032 | 64 | **2,032** |
| per-time-step (all combo variants) | one target month | 144 | 4 | **36** |
| per-time-step, batch 2 (`gru`, `mamba_nd`) | one target month | 144 | 2 | **72** |

**Why it matters.** The headline is that flat Mamba-2 (0.0493) beats everything
multidimensional. Every multidimensional arm sits in the 36-steps/epoch column,
under-trained relative to the flat arms it loses to.

**Not affected:** "axial vs flat cross-attention" is combo-vs-combo, so that
comparison is clean on this axis. The confound is specifically
*combo vs `embeddings`/`onehot`* — which is the headline comparison.

**RESOLVED — by aligning the batch upward, not by touching the learning rate.**

Scaling the combo LR was the obvious fix, but it leaves two models running
different hyperparameters, which then has to be argued for. Raising the flat
batch to meet the combo one needs no argument at all:

| | batch | steps/epoch | targets per step |
|---|---|---|---|
| flat (`embeddings`, `onehot`, flat s4) | **903** | 144 | 903 |
| every combo arm, incl. `mamba_nd` | **1** | 144 | 903 |
| `s4nd_2d_state`, `s4nd_2d`, `s4nd_4d` | **1** | 144 | 903 |
| `s4nd_3d` (per-flow samples, so 2/month) | **2** | 144 | 902 |

Identical batch, identical steps, identical `lr=1e-3`. Nothing is per-model.

**Why 903 and not some other number:** one combo sample structurally contains
all 903 kept combos, so the combo loader cannot go below 903 targets per step.
903 is therefore the *smallest* attainable alignment point — which also
minimises the large-batch generalisation penalty that now applies uniformly.

**The cost, stated plainly:** flat arms drop from 2,032 to 144 steps/epoch, so
they need far more epochs to reach the same number of updates. `epochs` is
raised to 5000 (a fuse, not a budget) and `patience` to 50 so early stopping,
not the cap, decides. Flat arms must be re-run; their configuration changed.

## S1-B. The S4 family validates twice per epoch

**Correction.** An earlier version of this audit claimed the S4 family trained
with no early stopping at all, because the `early_stopping` block in
`external/s4/configs/callbacks/checkpoint.yaml` is commented out. That was
wrong: `run_full_test.sh` adds it back on the command line
(`+callbacks.early_stopping.*`), `registry.py:26` maps the name to Lightning's
`EarlyStopping`, and `train.py` instantiates everything under
`config.callbacks`. Both sides had early stopping at patience 10.

What is real is `val_check_interval: 0.5`: the S4 family validated **twice per
epoch**. Lightning counts early-stopping patience in validation calls, so
patience 10 meant *5 epochs* there against *10 epochs* on the github side — the
S4 family stopped **earlier**, not later, the opposite of the original claim. It
did also get two checkpoint-selection chances per epoch instead of one, which is
a small advantage in the other direction.

Fixed: every s4nd experiment now sets `val_check_interval: 1.0`, and
`S4_COMMON` uses patience 50 with `min_delta` matched to the github side.

## S2-A. Batch size is mismatched inside the new matched pairs

The pairs added specifically to isolate *how the grid is injected* do not hold
batch size constant:

| pair | native ND | +ACA |
|---|---|---|
| S4 | `s4nd` batch **8** (hydra) | `s4+ACA` batch **4** |
| Mamba | `mamba_nd` batch **2** | `mamba+ACA` batch **4** |

A 2x batch difference is a 2x difference in optimizer steps per epoch (see
S1-A), so the pair comparison carried an optimization difference on top of the
architectural one.

**RESOLVED** by the S1-A alignment: both members of each pair now run at 903
targets per step and 144 steps per epoch.

## S2-B. GRU gets twice the optimizer steps of LSTM on identical variants

`COMBO_BATCH_SIZES` gave `gru` batch 2 and `lstm` batch 4 for the same
variants, so GRU took 72 steps/epoch and LSTM 36 — twice the updates for one of
two models in the same family.

**RESOLVED**: every combo entry is now 1, so the per-model differences are gone
along with the flat-vs-combo one.

## S2-C. Only the Transformer had its learning rate tuned

`config/models/transformer.yaml` is the sole model config overriding training:
`lr: 1e-4`, because 1e-3 diverges to NaN. The override is justified and
documented — but it means one model received LR tuning and the other ten run
at a single untuned default.

`gpt` is the sharp end of this: it is also a Transformer, it never got the same
treatment, and it runs at 1e-3. Whether GPT is merely stable or actually
well-optimized at that LR has not been tested.

## S3-A. Scheduler and gradient clipping differed between the two pipelines

| | github, was | S4 family, was | now (both) |
|---|---|---|---|
| plateau patience | 5 | 20 | **15** |
| plateau factor | 0.5 | 0.2 | **0.5** |
| gradient clipping | 1.0 | none | **1.0** |

Gradient clipping is matched at 1.0 on both sides. An earlier revision removed
it from the hydra pipeline, reasoning that `base.yaml` justifies clipping by the
Transformer's divergence and that S4's authors do not clip — so forcing it would
repeat the weight-decay mistake. That was wrong on two counts.

**Silence is not an instruction.** The weight-decay case rests on the s4 repo
*stating* that the kernel's decay "is always set to 0". The configs are merely
silent on clipping, and reading a prohibition into silence is over-
interpretation. Clipping is canonical for RNNs (exploding gradients) and
standard in Transformer and LLM-style recipes, which makes it protocol rather
than one model's private device.

**And it is not a no-op.** Measured over seed 26: `s4+ACA` exceeds gradient norm
1.0 in 1–3% of epochs with peaks near 41, and the Mamba family peaks at 52–173.
GRU and LSTM never reach 1.0 (max 1.5), so for them it is free insurance.
Leaving the hydra side unclipped would have split the `s4nd` vs `s4+ACA` pair —
the same S4Block, one side clipped and one not — which is precisely the
comparison the pair exists to isolate.

The scheduler change **is** protocol: once early-stopping patience moved to 50,
leaving the two pipelines at 5/0.5 and 20/0.2 would have meant different
annealing inside the same stopping window.

**RESOLVED**, and it uncovered a latent bug: `trainer.py` hardcoded
`factor=0.5, patience=5` in the `ReduceLROnPlateau` constructor, so
`scheduler_patience` / `scheduler_factor` in `base.yaml` were **never read**.
The values happened to match, which is why it was invisible.

That mattered once early-stopping patience moved to 50: with a plateau patience
of 5 the LR is halved ~10 times inside the wait window (1e-3 -> 1e-6), so the
tail of the wait trains at effectively zero LR and "patience 50" stops being a
convergence criterion. Plateau patience now tracks it at 15 — three reductions inside the window
(+15/+30/+45), ending at 1.25e-4, with 5 epochs left for the last one to act.
Both counters reset on improvement, so each drop is a rescue attempt: land one
and the full 50-epoch window restarts.

## S3-B. Determinism is stricter on the github side

`src/utils/seeding.py` sets deterministic algorithms, cuDNN deterministic mode,
disables the autotuner and sets `CUBLAS_WORKSPACE_CONFIG`. The S4 side calls
only `pl.seed_everything(seed, workers=True)` (`external/s4/train.py:728`).

This affects *reproducibility*, not expected performance — neither setting
advantages a model — so it is recorded rather than changed. Worth knowing when
a seed fails to reproduce exactly on the S4 side.

---

## Verified equal — no action needed

- **Capacity**: `d_model` / `hidden_size` = 128 and `dropout` = 0.1 across every
  neural model.

  > **CORRECTED 2026-09-03.** This line used to also claim "4 layers, `d_state`
  > = 64 across every neural model". Both halves are false.
  > **`d_state`:** the hydra S4/S4ND arms run `d_state: 128`
  > (`configs/experiment/s4*.yaml`) while `config/models/s4.yaml` runs 64, so
  > the `s4nd` vs `s4+ACA` pair — added specifically to hold the sequence model
  > fixed and vary only how the grid is injected — does not hold it fixed. The
  > Mamba pair (`mamba_nd` vs `mamba+ACA`) does, at 64 on both sides.
  > **RESOLVED 2026-09-13:** the whole S4 family now runs `d_state: 64` --
  > `config/models/s4.yaml` (unchanged from its original) and every hydra
  > S4/S4ND experiment config (lowered from 128), pinned by
  > `tests/test_protocol_config.py::test_s4_backbones_share_one_state_size`.
  > (An interim 2026-09-09 fix had raised the local side to 128 instead; the
  > author chose 64.) The recorded hydra runs were at 128, so their rows are
  > not comparable to the rerun on this axis either. The Mamba pair is 64 on
  > both sides, a different quantity that happens to share the number.
  > **Layers:** `config/models/mamba_nd.yaml` sets `depth_policy: coverage`,
  > which overrides `n_layers: 4` to `2 x (scan axes)` = 4 / 6 / 8 for the
  > 2d / 3d / 4d variants (484k / 713k / 949k parameters against ~485k for flat
  > Mamba). The recorded sweep ran at those depths. An iso-layer arm exists
  > (`depth_policy: fixed`) but has never been run.
  > Both are open protocol decisions, not code defects — they are recorded here
  > rather than silently changed.
- **Optimizer**: AdamW, `lr` 1e-3 (except Transformer), `weight_decay` 1e-4,
  `betas` (0.9, 0.999) — shared by both pipelines.
- **Epoch budget**: 5000 everywhere, with early stopping at patience 50 and
  `min_delta` 1e-4 on both sides — the cap is a fuse, not a budget.
- **Precision**: fp32 everywhere (`use_amp: false`, hydra `precision: 32`) —
  the fp16-vs-fp32 confound was already removed.
- **Loss**: `MaskedMSELoss` is documented to match per-series MSE scale (total
  squared error over valid slots / count of valid slots, averaged over the 2
  targets), so combo and flat losses are on the same scale.
- **Splits**: train <= 2023-12, val 2024, test 2025-01..05; normalization fit on
  the train slice only.
- **Selection metric**: `val_loss` / `val/loss` — both the normalized MSE that
  is also the reported objective.
- **Identity information**: fixed in PR #9; every multidimensional arm now sees
  exactly the identities that are not grid axes.

## Also verified since

- **Feature parity**: the s4 dataloaders call the github
  `TradeDataPipeline` directly (`trade_grid.py::_clean_pipeline`, same
  `input_len`, `lag_count=12`), so features, normalization and the 903-combo
  filter are shared by construction — they cannot drift.
- **Headline metric parity**: `norm_mse_combined` is
  `mean_squared_error(targ.reshape(-1), pred.reshape(-1))` on both sides
  (`scripts/inference.py:412`, `scripts/s4_eval.py:112`) — identical formula
  over both targets pooled.
- **Sample counts**: resolved by the S1-A alignment; every arm now runs 144
  steps/epoch at 903 targets per step.

## Not yet examined

- Whether the tree models (XGBoost / LightGBM) are comparable at all given the
  different paradigm — `n_estimators: 2000` was never tuned against the neural
  epoch budget.
- Weight-initialisation conventions across the two pipelines.
