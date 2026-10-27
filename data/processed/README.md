# Aggregated monthly research release

The canonical training, inference, baseline and S4/S4ND pipelines read
`hs2_monthly_aggregated.csv` directly under `fair-comparison-v5`.
Its 322,278 rows cover 1,542 state/commodity/flow series over January 2008
through May 2025. The experiment grid still starts January 2009.

Each State × Commodity × Import/Export × Time key occurs exactly once.
Value (USD) and Weight (kg) already sum all contributing countries and both
records in repeated source pairs. The loader performs no further duplicate
aggregation or missing-month filling. It rejects missing targets, duplicate
keys, incomplete calendars, or bytes that differ from the pinned fingerprint.
Exp 1 still intentionally sums these monthly series into its single target.

Imports retain the generic Vessel Value / Vessel SWT fields; exports retain
Containerized Vessel Total Exports Value / SWT. This is the existing
`legacy_mixed` target definition. Blanks and absent months were filled with
zero when building this release. SourceRows, MissingValueRows,
MissingWeightRows and DuplicateExtraRows are audit metadata, excluded from
model features. Aggregation does not establish that conflicting source
records were additive or validate official trade totals.

`dataset_manifest.json` records the original 16-file lineage and processing.
`duplicate_aggregation_audit.csv` records both selected values/weights and
their sum for every repeated pair. `config/data_manifest.json` pins the
monthly input actually consumed. The original CSV files are preserved for
the explicit legacy reproduction profile.

## Epoch steps

Equalization is within each experiment's common training calendar:

| Experiment / representation | Training samples per batch | Steps per epoch |
| --- | ---: | ---: |
| Exp 1 neural models | 33 aggregate windows | 4 |
| Exp 2 flat native and S4 models | 918 series targets from one month | 132 |
| Exp 2 native combo / S4ND full grids | 1 month | 132 |
| Exp 2 S4ND per-flow grid | 2 flow samples from one month | 132 |
| Exp 2 S4ND state-axis grid | 172 commodity/flow samples from one month | 132 |
| Exp 3 all window/lag cells | 918 series targets from one month | 108 |

Month membership determines actual training batches. Months are shuffled,
their target composition stays intact, and no training targets are dropped.
Evaluation batch sizes can differ without changing training steps. Exp 3's
common first target remains January 2015. Trees and statistical baselines
use the corresponding target set but do not have comparable gradient epochs.
Equal steps per epoch do not imply identical compute or equal total updates
after model-specific early stopping.

Data-only verification:

```
python scripts/prepare_exp2.py --check-data
python scripts/check_batch_alignment.py
cd external/s4
python ../../tests/check_s4_month_batches.py
```

The protocol version and consumed-source fingerprint changed. Old checkpoints
and results cannot be resumed or accepted as current under this release.
