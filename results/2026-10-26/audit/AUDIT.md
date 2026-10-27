# Numerical audit of the attached WCTR camera-ready PDF

Verified 26 October 2026 against the mounted Celsia disk. **The main results use the updated analyses and reproduce, but the paper is not entirely correct as written.** There is a material S4ND dimensionality-label mismatch, four minor rounding discrepancies, and several methodology/interpretation qualifications.

Audited PDF: `/tmp/codex-remote-attachments/01a17740-7f98-7720-abee-f67887000597/87E86639-0E82-4BBA-9BA2-C7D804C4C7DA/1-A84946CF_camera_ready_final.pdf`

SHA-256: `c591053555fcb60e8dea9ed67c7d5795765e83853150bf8fcd64e13ac5091b4e`.

## Corrections and qualifications

### 1. Figure 3: S4ND's 2D column does not represent the state-only 2D method

The printed S4ND improvements **+4.30% MSE / +1.01% MAE** reproduce the historical `s4nd|2d` arm. Its run contract sets the S4ND kernel `dim` to **3**. Its axes are **state × commodity-flow group × time**. The methods on page 5 describe the 2D representation as **state × time**, with commodity/flow encoded as features.

The distinct `s4nd|2d_state` arm implements that state-only layout. Its improvements are **+4.155531% MSE / +1.054127% MAE**, which round to **+4.16% / +1.05%**. These are not interchangeable arms. All 57 arms, including both S4ND variants, are included in the Table 2 ranks.

Correction options: show both layouts with explicit labels, or put the state-only arm in the common 2D column and explain the other layout separately. Do not silently rename the legacy arm or select between them by test performance.

Evidence: actual run contracts in `outputs/full/s4_s4nd_2d_seed26/metrics.json` and `s4_s4nd_2d_state_seed26/metrics.json`; `scripts/report_s4_experiments.py` in the disk's `box/autodl-tmp/wctr-shap/repo` explicitly documents the distinction.

### 2. Figure 2: four last-digit differences from full-precision means

Units below are normalized MSE × 100, as in the figure.

| Model | Window | Lags | Printed | Recomputed full precision | Correct rounding to 3 decimals |
|---|---:|---:|---:|---:|---:|
| LSTM | 24 | 24 | 4.947 | 4.946457 | **4.946** |
| Transformer | 6 | 12 | 4.920 | 4.920540 | **4.921** |
| Transformer | 6 | 24 | 4.888 | 4.888508 | **4.889** |
| S4 | 24 | 36 | 4.906 | 4.906514 | **4.907** |

The plot matches its saved, rounded CSV source. These differences are consistent with rounding that CSV before multiplying by 100 and formatting again. Regenerate annotations from the unrounded seed means. Rankings and highlighted minima do not change.

### 3. Methods: training calendar needs the sweep exception

The raw experiment grid starts January 2009, train ends December 2023, validation is 2024, and test is January–May 2025. However, **all 192 sweep runs explicitly use January 2015 as the first supervised training target**. The blanket statement that all models train from January 2009 needs this qualification. Input history and the first supervised target are different dates.

### 4. Methods: the targets are not uniformly containerized cargo

The consumed source uses `target_measure: legacy_mixed`: **imports use generic vessel value/weight; exports use containerized-vessel value/weight**. The source README explicitly documents this. Numerical reproduction does not support describing all targets as containerized cargo, nor establish that state-level international vessel trade is observed Mississippi barge flow. This affects interpretation, not the arithmetic checks below.

### 5. Methods: distinguish shared hyperparameters from equal parameter counts

Four layers, width 128, dropout 0.1, initial AdamW learning rate 0.0001 and weight decay 0.0001 match the neural run contracts (including nested S4ND configuration). But neural parameter counts vary substantially: the numeric aggregate arms span **171,906–797,314**, and full-panel arms span **180,326–966,018** parameters. If “all parameters nearly equal” means hyperparameter settings, say that explicitly. It must not imply parameter-count matching. Some SSM parameters also have weight-decay exemptions. A dropout of 0.1 is not a middle value between 0.1 and 0.2–0.5.

### 6. Interpretation: SHAP dollar values are magnitudes, not directional changes

The reported numbers reproduce: S4 cosine-month attribution **$4,173,070.64**, and LSTM other-state context attribution **$3,380,463.87**, averaged over the sampled 24 lanes × 5 months for the value output. These are **mean absolute net attributions relative to background predictions**, not signed shifts or causal effects of obtaining a feature. Suggested wording: “Mean absolute value-output attributions were approximately $4.2 million and $3.4 million per sampled lane-month, respectively.”

The code computes expected-gradients attributions using numerical integration. The plotted shares pool normalized value and weight attribution magnitudes; the dollar conversion uses the value output only. State this distinction if interpreting the percentages and dollars together.

### 7. Head comparison: qualify “significant reduction in LSTM performance”

The mean changes are correct: MSE **+272.0217%**, MAE **+68.4145%**. An independently computed two-sided exact paired Wilcoxon signed-rank test across the 25 seeds gives **p=0.00507164 for MSE**, but **p=0.41076517 for MAE** (uncorrected). The MLP is worse in 18/25 seeds for MSE and 12/25 for MAE. If “significant” is intended statistically, specify MSE, the test, and multiplicity treatment; do not imply both metrics show a significant deterioration.

## What passed

| Item | Verification |
|---|---|
| Table 1 | All **48 printed cells** match means from the October numeric-input runs at the displayed precision. |
| Figure 1 | All seven models' linear/MLP means and both metrics' percentage changes reproduce from individual runs; all 14 percentage annotations round correctly. |
| Figure 2 | All **64** cells trace to the 192 sweep runs, with three seeds per cell. **60** printed values match direct full-precision rounding; the four exceptions are above. |
| Figure 3 | Every printed improvement, encoder ratio, and dagger matches the underlying metrics/statistics, subject to the S4ND layout-label issue. Missing cells correspond to unplotted combinations, not zero performance. |
| Table 2 | All **10 ranks, 5 MASE values, and 8 non-baseline improvement percentages** reproduce. MASE was independently recomputed using each lane's 12-month seasonal-naive training scale. |
| Significance | Recomputed all **108** saved MSE/MAE comparisons from seed-averaged point losses, enumerating all 256 sign patterns across eight states. All p-values and BH-adjusted q-values match. |
| Figure 4 | All **10** displayed percentages reproduce from the saved attribution tensors; rankings, seed 26, 24 lanes and five months agree. |
| Cohort | Raw monthly source independently gives **1,542** initial combinations, **918** retained, **93** retained commodities, **8** states, **197** months, and density **61.6935% → 61.7%**. Filter uses nonzero raw **Value** on the training period. |
| Truth identity | Full-panel prediction populations and normalized truths agree across runs. The common full-panel truth also matches an independent reconstruction from the monthly source and training-only min/max scales within floating-point tolerance. |
| Seeds | Full-panel and sweep: **26, 105, 115**. Numeric/MLP learned aggregate models: **25** seeds. Deterministic aggregate baselines have one result each, not 25 independent fits. |

MAE, normalized MSE, RMSE and sMAPE were independently recomputed from prediction files for **577 runs**: 228 numeric, 178 MLP, and 171 full-panel. No mismatch above tolerance. RMSE is the mean of the separately computed value/weight RMSEs, then averaged across seeds; it is not generally the square root of the displayed combined mean MSE.

Specific narrative checks:

- LSTM aggregate MSE is **8.3815%** higher than Moving Average, correctly rounded to **8.4%**.
- S4 ASA 3D improves MSE by **6.5416267%**. Raw **p=0.0078125**, BH **q=0.151875**: the reported non-significance after correction is correct.
- Table 2 ranks reproduce: S4 1/15, Transformer FiLM 13/2, LSTM embeddings 12/1, XGBoost one-hot 49/5, Moving Average 52/43 (MSE/MAE).
- Moving Average is 52nd of 57 on ND MSE, correctly within the bottom fifth.
- Within-model sweep spreads `(max/min - 1) × 100` are LSTM **1.0837%**, Transformer **1.4235%**, S4 **1.0327%**, and bidirectional S4 **1.0603%**. “Within 1.5%” is correct when referring to each model's sweep, not the combined range across different models.
- Encoder ratios and their stated approximate range **0.991–1.018** match the displayed rounding.
- Embedding sizes **8+32+2=42** match run settings.

## External numerical claims

- **41%** drainage share is supported by [USGS](https://pubs.usgs.gov/circ/circ1133/introduction.html), referring to the conterminous United States.
- **About 400 million tons, 2015–2019**, and **92%** are present in [Amorim et al.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL104619). Keep the scope clear: 92% describes agricultural exports depending on the Mississippi River System, not the fraction of this study's trade data or a new 2026 measurement.
- **42,840 series, 12 aggregation levels, and the 2020 M5 competition** are supported by the [M5 accuracy paper](https://doi.org/10.1016/j.ijforecast.2021.11.013).
- **1,045 monthly M3 series** is supported by [Makridakis et al. 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194889).
- The qualitative statement about a meat-trade decline after the **2022** peak in COVID cases was not established by the supplied result artifacts. It requires a specified geography, COVID series, commodity aggregation and break analysis. Do not treat it as verified by this numerical audit.

## Sources, freshness and limits

The paper combines September corrected full/sweep results with October numeric-input, MLP-head and SHAP/statistical analysis. The old Git-tracked workbook is not its sole source. These results agree with the latest relevant analysis files located on the mounted Celsia disk, including the October 23 figure exports. This does not establish that no newer result exists on an uninspected remote machine.

Source roots:

- `.../WCTR2026/box/autodl-fs/wctr-exp1-numeric`
- `.../WCTR2026/box/autodl-fs/wctr-exp1-mlphead`
- `.../WCTR2026/local/wctr-h20-results/20260923T035254Z_pid36543/outputs/{full,sweep}`
- `.../WCTR2026/box/autodl-fs/wctr-shap-exp2/per_run`
- `.../WCTR2026/analysis/`

Machine-readable evidence: `checks.csv`, `figure-and-source-checks.csv`, `means.json`, `contracts.json`, `shap-check.json`, and `source-hashes.json`. **3,853 checks pass; four full-precision annotation-rounding checks fail.** This count includes supporting checks and is not a count of distinct paper claims. Semantic findings above are assessed separately.

This is a numerical/source-consistency audit, not a complete scientific-validity or citation sign-off. It did not rerun model training, exhaustively validate every checkpoint/completion receipt, establish that original country/duplicate aggregation is substantively valid, independently recreate SHAP through fresh model inference, or verify every bibliographic date/page number and historical narrative. Sweep cells were reconciled to the individual metric records; their prediction files were not all independently recomputed. These limits do not erase the confirmed label/methodology issues.

The submitted PDF, manuscript source, experiment results and workbooks were not modified.
