# Results synchronized from the Celsia disk on 26 October 2026

This snapshot combines the corrected H20 results with the October analyses used
in the camera-ready manuscript. **The latest aggregate results are the October
numeric-input results, not the aggregate sheet in the September workbook.**

| Content | Location | Runs / scope |
|---|---|---|
| September comprehensive workbook | `results.xlsx` (not in the public release; listed in the manifest with its SHA-256) | 816 run records; original export, 28 September |
| Latest aggregate numeric/linear-head results | [Summary](summaries/aggregate-numeric-october/summary.csv), [per-run table](summaries/aggregate-numeric-october/per_run.csv) | 225 learned runs + 3 deterministic baselines |
| October MLP-head comparison | [Summary](summaries/aggregate-mlphead-october/summary.csv), [per-run table](summaries/aggregate-mlphead-october/per_run.csv) | 175 learned runs + 3 baselines |
| Corrected full-panel results | [Run files](runs/full) | 57 arms × 3 seeds = 171 runs |
| Corrected window/lag sweep | [Run files](runs/sweep) | 64 cells × 3 seeds = 192 runs |
| September aggregate results underlying workbook | [Run files](runs/aggregate-september) | 453 runs; preserved separately |
| Numeric aggregate metrics and predictions | [Run files](runs/aggregate-numeric-october) | 228 runs |
| MLP aggregate metrics and predictions | [Run files](runs/aggregate-mlphead-october) | 178 runs |
| October source figures and analysis CSVs | [Analysis](analysis) | Original exports retained |
| Published LSTM/S4 SHAP panel tensors | [Attribution arrays](shap) | Seed 26; 24 lanes × 5 months |
| Numerical audit and outstanding corrections | [Audit](audit/AUDIT.md) | Checked against the attached camera-ready PDF |
| File checksums and source provenance | [Manifest](manifest.json) | SHA-256 and byte count for each synchronized file |

These are separate experiment populations; do not sum their run counts and
describe them as one benchmark. All run folders include `metrics.json` and
`predictions.csv`. Model checkpoints, training environments, and complete
completion-receipt bundles are not included in this results sync.

## S4ND variant identity: important for Figure 3

| Metric identity | Convolution axes | MSE / MAE improvement over Moving Average |
|---|---|---|
| `s4nd`, variant `2d` | State × commodity-flow group × time (three axes) | +4.30% / +1.01% |
| `s4nd`, variant `2d_state` | State × time (two axes) | +4.16% / +1.05% |
| `s4nd`, variant `3d` | State × commodity × time, flows handled separately | +3.89% / +1.40% |

The original paper figure puts the first row under “2D”, although the methods
describe the second row. Both runs are present here. The original figure
exports have not been edited; consult the audit before reusing them. The
headline +6.54% result belongs to **S4 ASA 3D**, a different model, and is
unaffected by this label mismatch.

## Provenance and verification

The imported monthly dataset matches the repository's
`data/processed/hs2_monthly_aggregated.csv` byte-for-byte. It uses mixed vessel
targets: generic vessel imports and containerized vessel exports. See
the data README for source aggregation limitations.

The 26 October audit recomputed prediction metrics for 577 runs, reconciled
all 64 sweep means, reproduced Table 2 ranks and MASE, checked the figures,
recomputed the 108 saved significance comparisons and BH corrections, and
recovered the SHAP attribution shares and dollar magnitudes. It found four
minor heatmap rounding discrepancies and the substantive S4ND labeling issue.
This does not constitute a full rerun, checkpoint audit, or publication sign-off.

The September workbook was preserved byte-for-byte in the private source
repository, with October results alongside it rather than silently changing old
experiment definitions within its sheets. The public WCTR2026 release omits the
workbooks and the historical pre-correction `graphs/` and `outputs/` trees; every
run behind the paper is here as `metrics.json` + `predictions.csv`.

`manifest.json` uses disk-relative source paths. Original metric payloads and
audit records retain their recorded machine paths as provenance; those paths
are not required to read the synchronized CSV/JSON results.
