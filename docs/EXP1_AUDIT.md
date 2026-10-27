# Experiment 1 code and leakage audit — 2026-09-05

Status: **open pending real CUDA/Triton Mamba verification**. This is an audit
of the current working tree, not certification of historical conference scores
or a claim that the source reports are independently validated ground truth.
No replacement dataset was downloaded or installed by this review. No full
paper training sweep was launched.

## Scope and frozen research choices

- Original 16 CSVs, eight states, and 97 represented HS2 chapters are preserved.
  The reference commit is `c5ce1b9db40a50989024ef77ed7b138037f21354`.
- The historical target mapping is total-vessel imports plus containerized-vessel
  exports. The explicit, source-pinned `legacy_sum` policy retains repeated
  source keys. It reproduces the old aggregation; it does not resolve ambiguous
  Arkansas export records. See [the source audit](LEGACY_DATA_AUDIT.md).
- Exp1 sums all series into one **eight-state aggregate**, not a U.S. national
  total. It skips the per-series sparse-cohort filter.
- Grid: January 2009–May 2025. Train ends December 2023; validation is 2024;
  test is January–May 2025. The corrected window path emits 132/12/5 targets.
- The manifest specifies 19 arms: six neural/SSM and two tree families, each
  with embeddings/one-hot and 25 seeds, plus three deterministic baselines once
  each: **403 expected run directories**, not 403 independent test datasets.

## Requirement-to-evidence map

| Requirement | Current implementation / evidence | Result |
|---|---|---|
| Preserve old data and duplicate behavior | `config/data_manifest.json`, `tests/test_legacy_reproduction.py`: exact raw values, monthly grid and historical cohort against the reference loader; raw Git comparison | Verified; source ambiguity remains disclosed |
| Fixed state/date/measure population | `DataConfig`, manifest domain guards, `test_exp1_data_integrity.py` | Verified |
| No future-fitted normalization | Aggregate before train-only min/max; future-target perturbation leaves train/validation values and features exactly unchanged | Verified |
| Targets excluded from their own inputs | Neural windows end at t−1; native tree extraction uses t−1 features; October mutation cannot alter October inputs in either encoding | Verified |
| Same target dates across families | `tabular_align_targets`, 132 training targets, 12 validation, five test; actual CLI round trips | Verified on CPU arms |
| No silent dropped/overweighted final batch | Exp1 batch size 33; four equal updates; target-weighted train/validation losses | Verified |
| Validation-only model selection | Training entry point discards test loader; neural minimum validation MSE and tree validation sets; nonfinite losses rejected | Reviewed and tested |
| Real models, not substitutes | Registered LSTM/GRU/Transformer/GPT/S4 and native trees; S4 genuine kernel CPU run; Mamba vendored-source guard | CPU arms verified; Mamba CUDA pending |
| Correct attention boundaries | GPT prefix causality and Transformer batch-isolation tests; attention sees only observed history | Verified |
| Correct metric domains/population | Normalized MSE/MAE; physical-domain percentage metrics; training-only pre-burn-in seasonal MASE/RMSSE scales; prediction recomputation and population checks | Verified |
| Truth tied to original CSVs | `src/utils/exp1_truth.py` reconstructs canonical targets and inverse transforms; forged self-consistent targets rejected | Verified |
| Checkpoint/config/source identity | Embedded and adjacent run contracts; native bundle hashes; inference mismatch guards | Verified |
| Completed training, not best-so-far | `exp1_completion.py` in inference, skip gate and collector; neural early-stop receipt and native estimator hashes | Verified |
| Faithful accepted resume | Optimizer/scheduler/RNG/early-stop state; rejected-resume logs untouched; unchanged-best extended run preserves weights and original selection contract | Verified |
| Honest runtime metadata | Unavailable requested neural device rejected; aggregate TF32 policy explicit | Verified by device mocks and CPU execution; CUDA arithmetic not measured here |
| Exact paper membership and failure propagation | Manifest roster, canonical contracts, failed training suppresses inference/workbook; launcher tests | Verified |
| Stop training descendants on interruption | Worker process groups, TERM then bounded KILL, reentrant-signal protection; disposable shell/training-child tests | Verified locally for normal and TERM-ignoring children |

## Important interpretation limits

Evaluation is rolling one-step-ahead: February can use January's observed
actuals. It is not a five-step forecast from a fixed December origin. The
backtest does not model Census publication delays or historical revisions.

Tree and neural arms share target dates, but not identical history capacity:
trees use their last-observation row plus lag features; neural models use the
full window. Architecture-specific histories must be described rather than
represented as identical inputs.

In Exp1, state/commodity/flow identities are all `ALL`. Consequently, comparing
embedding and one-hot arms here is not evidence about generalization across
categories. Their constant categorical representations still change model
parameterization.

There are only five test months. Repeating optimization with 25 seeds measures
training variability on those same months; it does not create 125 independent
test months. Seed control also does not establish bitwise identity across GPU
hardware or library versions; deterministic algorithms are requested with
warnings for unsupported operations.

Keeping the raw dataset fixed does **not** make corrected code reproduce the
conference's optimization, rankings, or checkpoints. The separate historical
profile reproduces data semantics/cohort selection, not every old code defect.

## Verification record and outstanding work

- The latest broad regression run before the interruption change passed **363
  tests**, excluding the two end-to-end test files. After that shell-only fix,
  all **10 launcher tests** passed, including repeated termination and a child
  that ignores TERM. Shell syntax and `git diff --check` passed.
- CPU train/reload/source-bound checks passed for both variants of LSTM, GRU,
  Transformer, GPT, S4, XGBoost, and LightGBM, plus all three baselines. Fresh
  targeted round trips were rerun after completion/resume changes; see the
  chronological verification notes in [LEGACY_DATA_AUDIT.md](LEGACY_DATA_AUDIT.md).
- These are short integration checks in temporary output directories, not full
  paper results. Canonical gates reject the shortened neural recipes.
- The two real Mamba cases exist in `test_exp1_neural_end_to_end.py` and are
  explicitly skipped on this Mac because CUDA is unavailable. Do not replace
  the mixer with a CPU stub to claim this requirement has passed.

On an authorized, configured CUDA/Triton host, run:

```bash
python -m pytest tests/test_exp1_neural_end_to_end.py -k mamba -q
```

Review the actual training/reload/evaluation results and resolve any failures
before closing this audit. No paid cloud job or remote-machine access is
authorized merely by this command being documented.
