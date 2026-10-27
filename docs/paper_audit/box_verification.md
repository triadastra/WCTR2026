# Training-machine copy vs. results snapshot

On 27 October 2026 a fresh read-only copy of the October experiment folders was taken from the GPU training machine and checked against [`results/2026-10-26/manifest.json`](../../results/2026-10-26/manifest.json). The check used `tools/verify_against_manifest.py`.

| Item | Value |
|---|---|
| Folders copied | `wctr-exp1-numeric`, `wctr-exp1-mlphead`, `wctr-shap-exp2`, excluding checkpoints and plot images |
| Files on the machine after excludes | 8,234 |
| September H20 full-panel / sweep folder | not present on that machine; checked through the snapshot instead |

## Manifest check (SHA-256 of each file)

| Group | MATCH | DIFFERENT | MISSING |
|---|---:|---:|---:|
| Aggregate numeric runs | 459 | 0 | 3 |
| Aggregate MLP-head runs | 359 | 0 | 3 |
| SHAP attribution tensors | 2 | 0 | 0 |

- **The 6 missing files** are the derived `numeric-summary/{per_run.csv, summary.csv, summary_metadata.json}` tables. They no longer exist on the machine, and their snapshot copies agree with means recomputed from the runs to within 1e-15.
- **Sanity hashes:** run-level `metrics.json` + `predictions.csv`, listed with Linux `sha256sum` and sorted with `LC_ALL=C`:
  - numeric: 456 files, `bf11d3960be4ef16b977faf1d623572d0927dc57a1d2affdbe69997e62185bef`
  - MLP-head: 356 files, `ec0711357b49467684cc2e350185d732145b361438833166fd485af8396d4a17`
- **Newer runs:** none. The machine holds no run folders beyond those in the manifest; the other `metrics.json` files are 400 per-epoch `logs/` training histories. The newest file is dated 2026-10-20, before the snapshot.
