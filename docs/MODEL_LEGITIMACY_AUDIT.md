# Model legitimacy audit — 2026-09-08

Historical audit before the subsequent implementation changes. See
[MODEL_IMPROVEMENTS.md](MODEL_IMPROVEMENTS.md) for fixes to categorical attention,
LightGBM subsampling and shared utilities, and the new identity-aware experiment.
The Mamba-ND qualifications and stale-results warning still apply.

Verdict: the active forecasting backbones are real implementations, but a blanket claim that every variant faithfully implements its name is not justified. The generic categorical attention is degenerate; Mamba-ND is a custom axial Mamba-2 adaptation; existing results are explicitly quarantined as stale.

Scope: source review of all 18 Python files in `src/models`, plus the active vendored Mamba-2 and S4/S4ND implementation paths, Mamba-ND reference scan code, relevant configuration and training dispatch. Other vendored research models received syntax checks only, not a full semantic audit. This assesses implementation identity, not authorship, licensing clearance, checkpoint authenticity, or empirical forecast quality. No model source was changed.

## Findings

1. **High — generic categorical “cross_attention” does not perform meaningful attention.** `encodings.py:366` and `:501` construct just one categorical key/value token. Softmax over one key is always 1, leaving query/key projections with zero gradient and reducing the attention output to a projected categorical vector (with dropout when enabled). The FiLM modulation is real, but its attention component has the same defect. `attention.py` already warns about it; a warning does not correct the mechanism. `make_transformer` and `make_mamba` still fall through to these encoders for the plain `cross_attention` and `film_attention` variants. Restrict these factories to supported variants or redesign the categorical tokenization before calling these arms attention. This finding does **not** apply to LSTM/GRU combo CA/FiLM: their factories replace the generic encoder and use real cross-group attention from `combo_attention.py`.
   **Status 2026-09-09: resolved by redesign.** Commit 5f1abee rewrote `CrossAttentionEncoding` / `FiLMAttentionEncoding` to three per-category key tokens with numeric-timestep queries (`architecture_version = categorical_attention_v2`; see `MODEL_IMPROVEMENTS.md`), so W_q/W_k now receive gradient. Consequence for reporting: the `transformer_cross_attention` / `transformer_film_attention` rows in `results.xlsx` (847k / 836k parameters) were produced by the degenerate single-key encoder and are **not comparable** to any rerun under the same arm name; `RERUN.md` records this.

2. **High for reproduction claims — `mamba_nd` is not an exact official Mamba-ND reproduction.** `_scan_axis` at `mamba_nd.py:109–151` folds other axes into the batch and runs separate sequences of length A, resetting state between axis lines. It also removes absent positions before scanning. The vendored official video block at `external/Mamba-ND/video_classification/src/mamba.py:227–234` supports flattening multiple dimensions into one ordered sequence; its default flattens all four. The local version additionally uses Mamba-2 instead of Mamba-1 and a different default schedule. These are material changes to state propagation, neighborhood distances and block design, beyond simply removing dependency wrappers. Describe it as a custom axial Mamba-2 model inspired by Mamba-ND; do not claim exact upstream equivalence. Reference: [official Mamba-ND repository](https://github.com/jacklishufan/Mamba-ND) and [paper](https://arxiv.org/abs/2402.05892).

3. **Medium — default Mamba-ND dimensionality comparison also changes directional coverage.** With four layers and the configured cyclic schedule, 2D scans Time/State in both directions; 3D scans Time both ways but State/Commodity forward only; 4D scans every axis forward only. Thus `bidirectional: true` does not mean all axes are bidirectional. The code warns and provides a coverage policy, but this remains a confound in interpreting dimensionality effects. Label the fixed-depth protocol explicitly and treat 4/6/8-layer full coverage as a separate capacity-changing ablation.

4. **Medium — LightGBM's configured row subsampling is inactive.** `lightgbm.py:58–85` sets `subsample=0.9` but never supplies `subsample_freq`. The library default is zero, disabling row bagging. Even supplying `subsample_freq` through the wrapper's extra kwargs would currently discard it. Expose and pass a positive frequency if 90% row sampling is intended. This remains a real LightGBM model, but the stated hyperparameter does not produce its apparent behavior. [Official parameter documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html#bagging_freq).

5. **High for empirical claims — model legitimacy does not validate the saved leaderboard.** `RESULTS_STATUS.md` explicitly marks the workbooks, graphs and `outputs/full` metrics as stale because their 903-series cohort was selected using the full timeline; the corrected cohort contains 918 series. `README.md:10` nevertheless says the benchmark is complete. Do not use the old artifacts to support corrected performance or ranking claims. This audit did not independently rerun training or authenticate every saved checkpoint.

6. **Lower-priority implementation edges.** `attention.PositionalEncoding` accepts dropout but never applies it. Both sinusoidal implementations assume an even hidden dimension. Flat LSTM/GRU allow bidirectionality but read `out[:, -1, :]`, whose backward half is not the backward final state summarizing the whole window; use the final hidden states if that is the intended bidirectional pooling. The configured unidirectional recurrent models are unaffected. All-masked inputs to the plain combo/latent attention softmax can produce NaNs; the axial helper handles dead grid lines separately. These do not make the standard configured backbones fake.

## File-by-file verdict

| File | Verdict and evidence |
|---|---|
| `lstm.py` | Real `nn.LSTM` with projection and regression head; combo and axial variants genuinely mix groups then recur over time. Correct group/time permutation is present. Bidirectional pooling caveat above. |
| `gru.py` | Real `nn.GRU`; equivalent legitimate combo/axial structure and pooling caveat. |
| `transformer.py` | Real pre-norm `nn.TransformerEncoder`, positional encoding and last-step head. Its “complete the architecture” header is stale, not evidence of an empty implementation. Plain categorical attention variants need qualification. |
| `gpt.py` | Real causal self-attention, pre-norm residual blocks, MLP and regression head. A GPT-style forecasting model trained from scratch, not an OpenAI model or pretrained language model. Axial variants exist despite stale introductory text. |
| `mamba.py` | Calls vendored `Mamba2`; genuine selective SSD scan. The disabled optional convolution uses the upstream PyTorch fallback; the stubbed Mamba-1 CUDA module is not the active Mamba-2 mixer. GPU execution unverified here. |
| `mamba_nd.py` | Real trainable multidimensional Mamba-2 architecture, but custom axial adaptation, not exact official Mamba-ND. Variant names containing `cross_attention` select scan dimensions here; they do not add attention. |
| `s4.py` | Loads standalone vendored `S4Block`; its FFT convolution selects the DPLR SSM kernel by default. Genuine S4, not an LSTM substitute. Axial attention plus S4 is a hybrid, distinct from S4ND. |
| `xgboost.py` | Real `XGBRegressor` pair, validation early stopping and native categorical handling. |
| `lightgbm.py` | Real `LGBMRegressor` pair and callback-based early stopping; row subsampling caveat above. |
| `random_forest.py` | Real sklearn `RandomForestRegressor` pair. Integer-ID variant uses ordinal splits, not learned embeddings or native unordered categorical splits. |
| `baselines.py` | Legitimate deterministic seasonal-naive, random-walk and moving-average formulas. Seasonal index `-12` is correct for predicting the step immediately after the window. Empty build method is intentional. |
| `combo_attention.py` | Real scaled dot-product cross-group attention, latent bottleneck attention, group-conditioned FiLM and masked axial mixing. This is custom architecture; source comments claiming exact paper-code parity were not independently proven. |
| `encodings.py` | One-hot and learned embeddings are genuine. Generic CA and FiLM-attention contain the single-key degeneration described above. |
| `attention.py` | Real attention primitives and FiLM arithmetic; single-key degeneration arises from caller tokenization. Positional/dropout and masking edges above. |
| `scan_schedule.py` | Functional paired, cyclic and legacy schedules; coverage depends on layer budget. |
| `tabular_common.py` | Builds previous-row features against next-row targets, with split/target-start filtering. Tree “embeddings” means IDs, not learned embeddings. Trees use the last feature row including lags, not the complete neural input window. |
| `base.py` | Legitimate abstract interface, factory and encoder setup; eagerly creates FiLM projections before optimizer construction. Abstract `pass` statements are intentional. |
| `__init__.py` | Registers real models with defensive optional imports. Missing dependencies remove models with warnings; no active LSTM fallback pretending to be Mamba/S4 was found. |

S4ND is implemented outside this directory: `external/s4/src/models/sequence/modules/s4nd.py` builds one SSM kernel per dimension and combines their Fourier kernels by tensor products/contraction, then applies inverse FFT. The 4D experiment uses the generalized `contract_version: 1`. This is a genuine multidimensional S4 implementation; the trade adapters are project-specific. Informer and the former project-level DSS stub are absent from the current factory.

## Verification and limits

- Parsed 217 Python files across project models and vendored model trees: zero syntax errors; some upstream docstrings emit invalid-escape warnings.
- Executed the actual standalone schedule function for the default 2D/3D/4D configurations and six paired/cyclic full-coverage checks: passed.
- Available bundled Python lacks torch, pytest, XGBoost, LightGBM, sklearn and einops. No neural forward/backward, GPU kernels, tree fits, checkpoint loads or full test suite were executed. Static syntax success is not numerical verification.
- No claim of exact original-paper parity can be established solely from comments saying “verbatim”; it needs an identified source revision and numerical comparison.

Overall: trust the core model identities at source level, qualify the custom/degenerate variants, fix the documented configuration issues, and require corrected runs before trusting performance claims.
