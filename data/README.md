# Preserved experiment data

The active workflow reads `processed/hs2_monthly_aggregated.csv` directly,
with `source_format: monthly`, `target_measure: legacy_mixed`, and
`duplicate_policy: error`. See [the processed release notes](processed/README.md)
for complete preprocessing, batching and lineage details. The original 16
CSVs remain unchanged and are used only by the explicit legacy profile.

`config/data_manifest.json` binds the monthly CSV actually consumed.
`processed/dataset_manifest.json` preserves the original report hashes.
The eight-state scope, calendar and all-merchandise HS2 domain are unchanged.

The repository pin identifies the preserved repository snapshot; it does not
independently prove which bytes produced the conference results.
It is not, by itself, an independent receipt proving that those bytes were
freshly downloaded from Census. A replacement Census/Dropbox download must
therefore be reconciled against the schema and manifest before changing the
pin.

> **Ground-truth build is currently blocked.** `ArkansasV2.csv` contains 2,250
> duplicate groups (4,500 rows) at the complete retained report grain
> `(State, Commodity, Country, Import/Export, Time)`. The paired rows contain
> conflicting values, and none of the file's remaining columns identifies a
> dimension that distinguishes them. The other 15 source files have no such
> collisions. Summing, keeping the first row, or keeping the last row would be
> an unsupported choice, so `scripts/build_ground_truth.py` fails closed and
> no independently validated strict ground-truth artifact is checked in. This optional strict builder is **not**
> the active training prerequisite. The historical workflow explicitly retains
> and sums these records without claiming that this resolves their ambiguity.

## Frozen scope

- States (exactly 8): Arkansas, Illinois, Iowa, Louisiana, Minnesota,
  Mississippi, Missouri, and Wisconsin.
- Source coverage: 2008-01 through 2025-05.
- Experiment grid: 2009-01 through 2025-05 (197 months). The complete 2008
  source year is retained and fingerprinted but is intentionally outside the
  experiment grid.
- Train: through 2023-12; validation: 2024-01 through 2024-12; test: 2025-01
  through 2025-05.

These dates and states are also independently pinned in
`src/data/ground_truth.py`, so an ordinary config edit cannot silently expand
the benchmark.

## Source files

```text
data/
├── exports/
│   ├── ArkansasV2.csv
│   ├── Illinoisv2.csv
│   ├── Iowav2.csv
│   ├── Louisianav2.csv
│   ├── Minnesotav2.csv
│   ├── Mississippiv2.csv
│   ├── Missouriv2.csv
│   └── Wisconsinv2.csv
├── imports/
│   ├── Arkansas2.csv
│   ├── Illinois2.csv
│   ├── Iowa2.csv
│   ├── Louisiana2.csv
│   ├── Minnesota2.csv
│   ├── Mississippi2.csv
│   ├── Missouri2.csv
│   └── Wisconsin2.csv
└── ground_truth/                   # emitted only after source validation
    ├── exp1_monthly_targets.csv.gz
    └── manifest.json
```

Every raw CSV contains `State`, `Commodity`, `Country`, and `Time`. The
pipeline assigns `Import/Export` from the source directory rather than trusting
an optional raw flow label.

The active historical target mapping is:

- Imports: `Vessel Value ($US)` and `Vessel SWT (kg)`.
- Exports: `Containerized Vessel Total Exports Value ($US)` and
  `Containerized Vessel Total Exports SWT (kg)`.

Blank selected cells become zero under historical preprocessing; this is a
reproduction convention, not an independently verified interpretation.
The released monthly targets already sum repeated complete source keys and
distinct regional contributions. The loader rejects any repeated monthly
key. The original `legacy_sum` policy remains available in the isolated raw
report reproduction profile.

## Optional strict ground-truth builder (not the active workflow)

The following describes a separate total-vessel builder, not the mixed-measure
dataset consumed by the current experiment launchers. It remains blocked on
the original source. Do not replace source files or change their pin to unblock
it without a new, explicit decision about the research dataset.

After the source collision is corrected,
`ground_truth/exp1_monthly_targets.csv.gz` will be the human- and
machine-readable target dataset. It has exactly these columns:

```text
State,Commodity,Import/Export,Time,Value,Weight
```

It is derived without fitted statistics:

1. Parse and validate the exact source bytes in `config/data_manifest.json`.
2. Require uniqueness at the retained source-report grain of state, commodity,
   country, flow, and month. An unexplained duplicate blocks the build.
3. Sum `Value` and `Weight` across distinct `Country` labels sharing
   `(State, Commodity, Import/Export, Time)`. Repeated target keys are additive
   only when their Country differs; identical report-grain keys are not
   silently deduplicated or summed.
4. For every series identity observed at least once in the 2009-01…2025-05
   grid, emit all 197 months and fill absent source months with zero.
5. Sort by the four target-key columns and write deterministic UTF-8 CSV in a
   gzip stream with a zero modification timestamp.

This occurs **before** the 95% sparse-series filter, min-max normalization,
lag construction, and supervised window sampling. The companion manifest
records both the compressed-file SHA-256 and the canonical uncompressed CSV
SHA-256, along with the source fingerprint, exact scope, row counts, measure
mapping, and transformation boundary.

The detailed keys are intentionally retained for auditability. Experiment 1
then obtains its single eight-state aggregate series by summing `Value` and `Weight` over
all panel rows independently for each month; it does not change the panel's
eight-state or time scope.

After replacing the invalid source file and updating its reviewed fingerprint,
build or verify the artifact from any working directory:

```bash
python scripts/build_ground_truth.py
python scripts/build_ground_truth.py --verify
```

On the current snapshot the first command intentionally exits nonzero and
reports the ambiguous groups. Once clean source exists, `--verify`
reconstructs the panel in memory and fails if either checked-in artifact differs
byte-for-byte.

## Active model preprocessing (directly from preserved CSVs)

The shared modelling pipeline subsequently:

1. Audits repeated source keys and sums the selected raw targets by month and
   series under the explicit historical policy. Per-series experiments select
   the sparse-series cohort using training-period raw `Value` only; Exp1
   aggregates all series and skips that cohort filter.
2. Fits exact per-series min-max statistics on training data only.
3. Applies those statistics to train, validation, and test data.
4. Adds 12 monthly lags per target and sine/cosine month encodings.
5. Builds 36-month historical windows with chronological target splits.

Missing source months are zero-valued observations in the target panel. They
are not imputed using future values. Monetary values are USD and weights are
kilograms.
