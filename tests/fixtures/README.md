# Historical regression fixture

`legacy_dataloader_c5ce1b9.py.txt` is the byte-identical
`src/data/dataloader.py` from local historical commit
`c5ce1b9db40a50989024ef77ed7b138037f21354`.

SHA-256: `55b96c34bb69f131a3a1c419c1e48f1112b2cd93c48e312c8f1aa21ad64327f4`.

That commit is not reachable from current `main`. Keeping the exact source
here makes the legacy-equivalence regression runnable in a fresh or shallow
checkout, without downloading or relying on extra local Git history.

This is test-only historical code, not the active forecasting pipeline. Its
full-timeline cohort behavior is deliberately retained for the explicit legacy
comparison and must not be used for new benchmark training. The regression
checks the digest before executing this reference on the pinned raw reports.
