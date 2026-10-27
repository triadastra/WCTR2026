# Text fixes for the submitted PDF

The LaTeX source is not part of this release. The "old" strings are copied from the PDF text and may differ slightly from the LaTeX (hyphenation, macros). Ledger IDs refer to `claims_ledger.csv`.

## Blocking

**C390 (Abstract)**
- Old: `In high dimensions, statistical baselines do not outperform DL algorithms.`
- New: `In high dimensions, every deep-learning arm had lower MSE than the statistical baselines, although on MAE Moving Average beat 11 of 48 neural arms and no difference survived BH correction.`

**C419 (Sec. 6.2)**
- Old: `their MSE is about 6.5% lower than moving averages, though the difference is not significant after Benjamini–Hochberg correction.`
- New: `the best ND model (S4 ASA 3D) has about 6.5% lower MSE than Moving Average, though the difference is not significant after Benjamini–Hochberg correction.`

**C424 (Sec. 6.2)**
- Old: `These features shift the predicted value in the series by about 4.2 million USD per lane-month and 3.4 million USD per lane-month, respectively.`
- New: `Their mean absolute attributions to the value output were about 4.2 and 3.4 million USD per sampled lane-month, respectively.`

## Material

**C167 / C261 (Fig. 3, S4ND row, 2D column)**
- Old: MSE `+4.30`, MAE `+1.01`. These come from `s4nd|2d`, a 3-axis kernel over state × group × time.
- New, either:
  - MSE `+4.16`, MAE `+1.05`, from `s4nd|2d_state` (state × time, as defined in Sec. 4.3); or
  - keep the values and relabel the column for S4ND as the state × commodity-flow-group × time layout.

**C397 (Sec. 4.5)**
- Old: `We trained on data from 2009-01-01 to 2023-12-01, validated on 2024-01-01 to 2024-12-01, and tested on the first five months of 2025.`
- New: append `For the window/lag sweep (Part 2), the first supervised training target is January 2015; inputs still draw on history from 2009.`

**C403 (Sec. 4.1)**
- Old: `We obtained state-level trade data on containerized cargo along the Mississippi River`
- New: `We obtained state-level vessel trade data along the Mississippi River (generic vessel value/weight for imports, containerized vessel value/weight for exports)`
- Also adjust `containerized vessel value and containerized SWT (weight)` in Sec. 4.2 to match.


**C408 (Sec. 5.1)**
- Old: `We observed a significant reduction in LSTM performance, while other models performed slightly better.`
- New: `The MLP head worsened LSTM sharply (MSE +272%, paired Wilcoxon p=0.005 over 25 seeds; the MAE change was not significant, p=0.41), while the other six models improved by 6-36% in MSE.`

**C399 (Sec. 4.5)**
- Old: `For comparability, we have set all parameters to be nearly equal across the models.`
- New: `For comparability, we shared hyperparameters across models (4 layers, width 128, dropout 0.1, AdamW 1e-4); parameter counts still differ (about 0.18M-0.97M).`

**C386 (Abstract)**
- Old: `For wide or multidimensional series, 2D and 3D SSMs performed better on MSE, but 3D had worse MAE.`
- New: `For wide or multidimensional series, 2D and 3D SSMs performed best on MSE, but the best 3D SSM (S4 ASA 3D) ranked only 15th on MAE.`

## Cosmetic

**C073 / C081 / C082 / C108 (Fig. 2 annotations).** Regenerate the annotations from unrounded seed means:

| Cell | Old | New |
|---|---:|---:|
| LSTM, w24/l24 | 4.947 | 4.946 |
| Transformer, w6/l12 | 4.920 | 4.921 |
| Transformer, w6/l24 | 4.888 | 4.889 |
| S4, w24/l36 | 4.906 | 4.907 |

**C400 (Sec. 4.5)**
- Old: `(since Transformers mostly use 0.1 in prior work and RNNs use 0.2-0.5, this is a middle value)`
- New: `(0.1, the common Transformer default, for all models)`

## Optional disclosure

**Sec. 4.3:** after the 42-dimensional embedding sentence, add `The ASA-1D and FiLM variants use a 4-dimensional flow embedding (44 dimensions in total).`
