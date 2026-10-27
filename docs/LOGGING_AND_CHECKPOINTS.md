# Run logs and model checkpoints

The prepared H20 deployment links `outputs/` to
`/root/autodl-fs/wctr-20260922/production-outputs/`. Keep that link in place:
logs, checkpoints and evaluation outputs then live on the persistent volume.
Each experiment/model/variant/seed (and Exp 3 window/lag) has a separate run
directory under `agg/`, `full/` or `sweep/`. No full training starts merely by
installing these logging changes.

| Family | Files relative to its run directory |
| --- | --- |
| Native neural models, including modular S4 | `best.pth`, `last.pth`, `checkpoint_manifest.json` |
| External S4/S4ND | `hydra/checkpoints/val/loss.ckpt`, `hydra/checkpoints/last.ckpt`, `hydra/checkpoint_manifest.json` |
| XGBoost, LightGBM, random forest | `best_xgb_value.joblib`, `best_xgb_weight.joblib`, `best_xgb_metadata.json`, `checkpoint_manifest.json` |
| Parameter-free baselines | `baseline_definition.json`, `checkpoint_manifest.json` (no learned weights) |

`best` is validation-selected. `last` is the latest completed epoch and includes
optimizer/scheduler state for resuming; it is not necessarily the best model.
Native resume files also retain early-stopping, random-generator and history
state; Lightning manages its corresponding training state. Trees retain their
fitted estimators for inference, not incremental neural-style resume state.
The manifest lists relative paths, byte sizes, SHA-256 hashes and the run
contract. It is a completed-training inventory, not a substitute for the
existing source/truth/early-stopping completion gates. A resumed training
attempt can change checkpoint bytes; use its status and verify hashes before
treating an older inventory as current.

Native `logs/` contains the UTC text log, `history.csv`, `metrics.json`,
`status.json` and append-only `events.jsonl`. Each attempt has a distinct ID;
start/end/failure events survive fresh starts and resumes. Full Python
tracebacks are written to the text log for caught training failures. During
neural training/validation, batch-boundary heartbeats are throttled to 30
seconds. This is not a background timer: a blocked batch or a tree estimator
inside its one-shot fit cannot refresh it.

External S4/S4ND writes `hydra/logs/status.json`, `events.jsonl`,
`epochs.jsonl`, `resolved_config.json` and
`lightning/version_*/metrics.csv`, without requiring W&B. Epoch records contain
losses, learning rates, global steps and timing. CSV versions and attempt IDs
separate restarts. Caught Lightning exceptions include their traceback in the
event journal. `finished` is published after checkpoint inventory and training
completion evidence have been written; failed post-fit publication remains
non-finished and is diagnosed from the process log.

The canonical controller saves its combined output to
`outputs/pipeline_runs/<launch-id>/pipeline.log`, alongside its existing
manifest and environment snapshot. Python output is unbuffered and fault
handling is enabled. Per-run `train.log` and `inference.log` remain available.

Read-only live monitor:

```bash
python scripts/train_status.py --root outputs --watch
```

Native checkpoint writes and structured local records are flushed to disk;
checkpoint replacement is atomic. On POSIX, checkpoint, tree-bundle and
structured-record replacements also sync their containing directory. These
safeguards are not a backup and do not guarantee survival of storage loss.
SIGKILL, power loss and remote host
failure cannot reliably write a final failure event: the monitor detects stale
heartbeats (and dead local processes). Retain a separate backup before deleting
the server or persistent volume. Load pickle-based `.pth`, `.ckpt` and `.joblib`
files only from trusted sources.
