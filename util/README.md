# Utility layout

The canonical train/evaluate/build workflow lives in `scripts/`. Files directly
under `util/` are optional paper-analysis, figure, or artifact-maintenance tools.

`util/diagnostics/` contains manual environment and model checks. They are not
pipeline completion gates; automated regression coverage belongs in `tests/`.

Historical one-off builders and repair scripts are intentionally not retained:
Git preserves them, while keeping executable copies beside current utilities
makes it too easy to regenerate artifacts with an obsolete protocol.
