# Original-source field audit — 2026-09-13

The original 16 CSVs are preserved. Their manifest SHA-256 checks pass, with source fingerprint `e5c6610ff4d751fa18e313b06fed04b6c3846ee0d1776417b2b68af4f99a264d`.

Both flows contain containerized-value and containerized-weight columns. They also contain generic `Vessel Value ($US)` and `Vessel SWT (kg)` columns. The explicit total-vessel names required by the strict ground-truth builder are absent: imports do not have `Vessel Customs Value (Gen) ($US)` / `Vessel SWT (Gen) (kg)`, and exports do not have `Vessel Total Exports Value ($US)` / `Vessel Total Exports SWT (kg)`. The generic labels alone do not establish an audited mapping to those strict names.

| Flow | Raw field | Nonblank numeric rows | Blank rows | Selected by current `legacy_mixed` target |
|---|---|---:|---:|---|
| Imports, 443,184 rows | Vessel Value ($US) | 443,184 | 0 | Yes |
| Imports | Vessel SWT (kg) | 443,036 | 148 | Yes |
| Imports | Customs Containerized Vessel Value (Gen) ($US) | 427,126 | 16,058 | No |
| Imports | Containerized Vessel SWT (Gen) (kg) | 426,976 | 16,208 | No |
| Exports, 562,243 rows | Vessel Value ($US) | 562,243 | 0 | No |
| Exports | Vessel SWT (kg) | 562,229 | 14 | No |
| Exports | Containerized Vessel Total Exports Value ($US) | 505,641 | 56,602 | Yes |
| Exports | Containerized Vessel Total Exports SWT (kg) | 505,628 | 56,615 | Yes |

Every nonblank entry in these eight flow/field combinations parsed numerically; all parsed values were nonzero. Counts describe raw rows, including duplicated rows. The pipeline's historical blank-as-zero behavior is a declared preprocessing assumption, not proof that a blank report cell is an observed zero. Column availability does not by itself settle why the historical choice was made or the semantics of every missing entry.

The current benchmark therefore measures a mixed-scope trade-value/weight target. It must not be described as uniform container demand, container counts, or a uniform total-vessel measure. A uniform containerized profile could select fields present in both flows, but would change the target and potentially its cohort; that requires a separate experiment and an explicit treatment of the missing entries. No target columns were switched in this fix.

## Duplicate-source ambiguity

The loader reports 2,250 duplicate report-grain groups / 4,500 rows, with 2,198 groups containing conflicting selected values. The key is state, commodity, country/region, flow, and month. The files carry no extra dimension explaining how the paired reports differ. The current pinned `legacy_sum` behavior sums them for historical reproducibility. Keeping first/last, taking a maximum, or averaging them would introduce a different unsupported assumption.

The strict ground-truth path continues to reject ambiguous sources. Resolving this scientifically requires the original report provenance or a separately validated replacement source; code cannot infer whether those records are additive. Workbook provenance now carries the duplicate audit and this evidence boundary rather than implying that hash agreement certifies source truth.

## Evaluation scope

The January–May 2025 test values have already been used during development. Relabeling them, adding optimization seeds, or changing a p-value formula cannot make them untouched evidence. Current reports therefore describe retrospective results and distinguish optimization variability from variation across forecast origins.

The backtest uses observed prior-month values. Source release vintages and availability dates have not been established. It is not certified as a real-time forecasting exercise or a five-step prediction issued at one fixed December origin. A future confirmatory study must freeze source definition, model/tuning choices, comparison families, and stopping rules before inspecting its holdout; operational claims additionally need availability-aware inputs.
