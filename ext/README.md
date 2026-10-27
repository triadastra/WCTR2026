# `ext/` — significance & Diebold-Mariano tooling for the Exp 2 benchmark

Read-only analysis code over `outputs/full/`. It computes the statistical
statements the headline per-combo benchmark rests on: which arms are genuinely
separated, and which are tied.

> This directory previously also held the exogenous-feature ablation ("Exp 4").
> That experiment was descoped and has been removed; only the significance
> tooling below — which the Exp 2 analysis imports — remains.

## Hard constraint (do not violate)

> **Do not remove or modify the original working tests and model files.**

Nothing here edits `src/models/`, `scripts/train.py`, `scripts/inference.py`,
`external/s4/train.py`, the existing S4 dataloaders, the existing configs, or
`outputs/{agg,sweep,full}`. Existing code is **reused by import**, never copied
or changed. A quick guard:

```bash
grep -rl "outputs/full\|outputs/agg\|outputs/sweep" ext/ | xargs grep -l "open(.*['\"]w" 2>/dev/null
# -> must be empty: ext/ never opens an original output for writing
```

## Layout

```
ext/
  README.md                  this file
  eval/
    resampling.py            significance tests at the combo unit -- paired t (Holm),
                             cluster bootstrap, Friedman/Nemenyi. Imported by
                             util/make_{significance,vs_baseline,cd_diagram,axial_vs_flat}.py
    diebold_mariano.py       DM over outputs/full/<run>/predictions.csv: every arm
                             vs the PRE-REGISTERED moving-average baseline
                             (seed-averaged, series-clustered, Holm-corrected),
                             plus a descriptive all-pairs matrix. The earlier
                             version's test-selected "champion" comparator is
                             removed (selection-on-test).
    s4_dump_preds.py         dump S4/S4ND per-series preds (monkeypatch reuse of s4_eval)
  outputs/                   generated, gitignored
    dm/                      DM artifacts (csv + heatmap)
    s4_preds/                dumped S4/S4ND predictions
```

## Unit of analysis

The unit is the **combo**, not the (combo, month) point. The test set is 918
trade lanes x 5 months = 4,590 points under the corrected train-only cohort
(903 x 5 = 4,515 under the legacy full-timeline cohort), but those are not
independent
observations: the five months of one lane are serially correlated, and lanes in
the same month share seasonal and macro shocks. Treating them as independent
shrinks the standard error by roughly sqrt(5) and inflates significance by three
to four orders of magnitude. `resampling.py` documents this in full; it is why
the combo-clustered result is the primary one everywhere.
