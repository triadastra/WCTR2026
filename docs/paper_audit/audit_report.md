# Paper claim audit: *Multidimensional Demand Forecasting using State Space Models* (WCTR 2026)

Audited 27 October 2026. Ledger: [`claims_ledger.csv`](claims_ledger.csv). Regenerate it with
`python tools/recompute_results.py && python tools/build_claims_ledger.py`.

| Side | Pinned version |
|---|---|
| Manuscript | Submitted PDF, SHA-256 `c591053555fcb60e8dea9ed67c7d5795765e83853150bf8fcd64e13ac5091b4e` (not redistributed here). Claims were transcribed from its text and from 150-dpi renders of its figures. |
| Results | [`results/2026-10-26/`](../../results/2026-10-26/), exported from the private source repository at commit `0c43935ac7d32bf212996a3bb1fefb442f43a2d9`. All 2,500 manifest files shipped here match their recorded SHA-256. |
| Raw data | [`data/processed/hs2_monthly_aggregated.csv`](../../data/processed/), SHA-256 `585abc646e62283dca0f27854d73d60814cdae6ab64b12e545f7a84f317287fb` |

**Method.** Every metric was recomputed from `predictions.csv` for all 769 runs: 228 numeric aggregate, 178 MLP-head, 171 full-panel and 192 sweep. All recomputed metrics agree with the stored `metrics.json` within 1e-14. From those runs I rebuilt seed means, improvements over Moving Average, ranks over 57 arms and encoder ratios. Other quantities were rebuilt from more basic sources:
- **MASE:** from the raw dataset.
- **Significance:** with an independent implementation of the state-clustered wild bootstrap (8 clusters, all 256 sign patterns), with BH applied separately to each metric's 54 comparisons.
- **SHAP:** shares and dollar values from the attribution tensors in `results/2026-10-26/shap/`.
- **Cohort counts:** from the raw dataset.

The analysis CSVs, summaries and the earlier audit in the results snapshot were not inputs. They were compared afterwards and agree.

Separately, a fresh read-only copy of the October aggregate and SHAP folders was taken from the training machine. It matched the manifest byte-for-byte (820 of 820 files; see [`box_verification.md`](box_verification.md)).

## Verdict counts

424 claims: **408 SUPPORTED**, 8 CONTRADICTED, 5 OVERCLAIMED, 1 MISCOMPUTED, 2 UNVERIFIABLE.
Severity of the 16 rows needing attention: 3 blocking, 7 material, 5 cosmetic, and 1 unrated (C404, unverifiable).

| Where | Supported | Not supported |
|---|---:|---:|
| Table 1 (48 cells) | 48 | 0 |
| Table 2: 23 values, 8 daggers, 4 best-variant choices | 35 | 0 |
| Fig. 1 (14 percentages) | 14 | 0 |
| Fig. 2: 64 cells, 4 highlighted minima | 64 | 4 (last-digit rounding) |
| Fig. 3: 94 values, 94 daggers, 20 encoder ratios | 206 | 2 (S4ND "2D" label) |
| Fig. 4 (10 shares) | 10 | 0 |
| Prose claims | 31 | 10 |

**Bottom line:** the reported metrics are correct. Every number in Table 1, Table 2, Fig. 1 and Fig. 4 reproduces exactly at printed precision, and so does every dagger and encoder ratio. The only numeric problems are four last-digit rounding slips in Fig. 2 and the S4ND "2D" column in Fig. 3, which shows a different layout from the one the methods describe. Everything else that needs attention is wording or scope.

## Blocking: abstract and conclusion wording

1. **C390, Abstract:** "In high dimensions, statistical baselines do not outperform DL algorithms."
   - This is true for MSE: all 48 neural arms have lower MSE than Moving Average, which ranks 52nd of 57.
   - It is not true for MAE: Moving Average ranks 43rd of 57 and beats 11 of the 48 neural arms.
   - No arm beats Moving Average after BH correction (smallest q = 0.084).
2. **C419, Sec. 6.2:** "ND models … their MSE is about 6.5% lower than moving averages."
   - 6.54% applies only to the best ND arm (S4 ASA 3D).
   - Across 28 ND arms, the median improvement is 4.14% (range 0.67%–6.54%).
3. **C424, Sec. 6.2:** "These features shift the predicted value … by about 4.2 million USD … and 3.4 million USD".
   - The magnitudes reproduce exactly: $4,173,070.64 and $3,380,463.87.
   - However, they are mean *absolute* attributions relative to the background, not signed shifts.

## Material

4. **C167 / C261, Fig. 3, S4ND "2D" column.**
   - The printed +4.30 (MSE) and +1.01 (MAE) reproduce `s4nd|2d`. Its run contract sets `model.layer.dim = 3`, meaning state × commodity-flow group × time.
   - Section 4.3 defines 2D as state × time. That layout is `s4nd|2d_state`, at +4.16 / +1.05.
   - Both arms are among the 57 ranked arms, so Table 2 is unaffected.
5. **C397, Sec. 4.5:** "trained on data from 2009-01-01". All 192 sweep runs (Part 2) set `train_target_start = 2015-01-01`.
6. **C403, Sec. 4.1–4.2:** "containerized cargo".
   - The targets are `legacy_mixed`: imports use generic vessel value/weight, and exports use containerized vessel value/weight ([`data/processed/README.md`](../../data/processed/README.md)).
7. **C408, Sec. 5.1:** "a significant reduction in LSTM performance, while other models performed slightly better".
   - LSTM MSE +272.0%: paired exact Wilcoxon p = 0.0051, with 18 of 25 seeds worse.
   - LSTM MAE +68.4% is not significant: p = 0.411.
   - The other six models improved MSE by 6–36%, which is more than "slightly".
8. **C399, Sec. 4.5:** "set all parameters to be nearly equal". Hyperparameters are shared, but neural parameter counts span 180,326–966,018.
9. **C386, Abstract (ambiguous):** "3D had worse MAE".
   - 3D beats 2D on MAE for S4, Mamba-ND and S4ND, and loses for S4-bidir and Mamba-2.
   - The sentence is only clearly true of S4 ASA 3D's rank: 1st on MSE, 15th on MAE.

## Cosmetic

10. **Fig. 2 (C073, C081, C082, C108):** four last-digit rounding slips.
    - LSTM 24/24: 4.947 → 4.946
    - Transformer 6/12: 4.920 → 4.921
    - Transformer 6/24: 4.888 → 4.889
    - S4 24/36: 4.906 → 4.907
    - [`figures/sweep_heatmaps.png`](../../figures/sweep_heatmaps.png) shows the corrected values.
11. **C400:** a dropout of 0.1 is the low end of 0.1–0.5, not a "middle value".

## Resolved by this release

- **C405, Sec. 5:** the paper's code link `https://www.github.com/triadastra/WCTR2026` returned HTTP 404 on 27 October 2026 because the repository did not exist yet. The results had been kept in a private source repository. This repository is that link.

## Unverifiable

- **C404:** the meat-trade decline after the 2022 COVID peak needs an external COVID series.
- **C386:** see item 9 above.

## Supported, but worth disclosing

- **Embedding width:** the ASA-1D and FiLM arms of GRU, LSTM and Transformer (18 runs, including Table 2's Transformer FiLM) use a 4-dimensional flow embedding, for 44 dimensions in total. The stated 42 applies to the flat arms.
- **"Within 1.5%":** holds within each sweep model (largest 1.42%), not across models (2.23%).
- **"Deep learning models performed better on this ND test split":** MSE only.
- **"1542 valid combinations":** includes 2 lanes that are zero throughout.
- **NVIDIA H20:** recorded by `cuda-preflight.json` for the October aggregate runs. The full-panel and sweep runs record only `cuda`.

## Snapshot completeness note

The `metrics.json` of 24 full-panel S4/S4ND runs names `predictions_norm.csv`, which was not synchronized. Their `predictions.csv` holds real units only. `tools/recompute_results.py` rebuilds the normalized values from training-period min/max per lane, a method that reproduces other runs' normalized columns to 4e-16. Their metrics then match exactly.

## Relation to the earlier snapshot audit

This pass independently reproduces all seven findings in [`results/2026-10-26/audit/AUDIT.md`](../../results/2026-10-26/audit/AUDIT.md). It adds the abstract's baselines-vs-DL scope (C390), the scope of Sec. 6.2's "ND models ~6.5%" (C419), the ambiguous "3D had worse MAE" (C386), and the repository link (C405). External facts (41%, 400 Mt, 92%, M5, M3) and the bibliography were out of scope here.
