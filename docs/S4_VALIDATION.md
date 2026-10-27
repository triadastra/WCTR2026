# S4 migration verification — 2026-09-20

## Directionality update — 2026-09-22

The canonical roster now pairs every 1-D S4 arm with a separately labelled
`s4_bidir` arm: Exp 1 flat encodings, Exp 2 flat encodings and axial variants,
and Exp 3's full window/lag grid. Causal `s4` outputs retain their existing
names. Both directions use the same data, seeds, optimization and selection
settings, starting at LR `1e-4` with the existing plateau scheduler and no
hyperparameter search. S4ND is unchanged.

This adds 113 runs: 50 in Exp 1, 15 in Exp 2 and 48 in Exp 3. The complete
S4-family inventory is now 238 runs / 50 configurations; the whole benchmark
contains 453 aggregate runs, 171 panel runs and 192 sensitivity runs.

For the flat embedding model with 28 numeric features, eight states,
93 commodities and two flows, measured trainable parameter counts are
**180,326 causal** and **213,094 bidirectional**. This is a directionality
control, not a parameter-matched comparison.

Checks exercise the real two-sided kernel, all window/lag shapes, native
forward/backward and checkpoint round trips, external numerical parity,
fresh-process construction and seeded initialization, separate result/figure
labels, paired contracts, and rejection of opposite-direction checkpoints.
All eight external synthetic training/early-stopping/evaluation lifecycles
passed. The source audit still finds ten byte-identical S4 core files; only
local experiment recipes changed (48 local additions now).

The native import-order issue is corrected by loading Lightning before the
temporary vendor namespace. No production experiments or GPU deployment
were performed for this update. The full-suite figures in the historical
migration record below describe the earlier, causal-only roster.

The updated complete suite finished with **706 passed, 2 skipped, 48 warnings**
in **918.76 seconds**. Both skips are the real CUDA/Triton-only Mamba CLI
checks; no S4 checks were skipped. This includes real-source aggregate CLI
round trips for both S4 directions and encodings, all eight external
synthetic lifecycles, and the previously implemented full-window tree inputs.
The controller's `--print-plan`, shell syntax checks, `git diff --check` and
the S4 source audit also passed. The expected full production roster is
**816 runs**, not completed results; GPU execution remains unverified.

## Outcome and limits

The current local working tree uses the authors' modular S4 backbone for every
canonical S4 arm. S4ND remains on its external modular implementation with the
existing, disclosed padding corrections. No existing vendored author-source
files were changed by this migration.

The complete pytest suite finished with **659 passed, 2 skipped, 40 warnings**
in 716.26 seconds. Both skips were the real Mamba CUDA/Triton CLI checks; no
S4 tests were skipped. This verifies the tested CPU paths, not an absence of
all possible defects or completion of the production experiment roster.

The supplied remote server was reachable, but `nvidia-smi` reported
**No devices were found**. H20/CUDA verification and deployment of these local
changes have not been completed. No production GPU experiments were launched.

## Requirement-by-requirement evidence

| Requirement | Evidence | Status |
|---|---|---|
| Every canonical S4 arm uses the external modular implementation | Actual forward-method source paths checked for all eight native variant names; Exp 2's six external S4/S4ND lifecycles exercised; no standalone fallback | Passed on CPU |
| Preserve genuine author code | Fresh official checkout at `e757cef57d89e448c413de7325ed5601aceaac13`; 10 designated core files byte-identical; 423 upstream files identical overall | Passed |
| Disclose local changes | 12 pre-existing modified upstream files documented in `external/s4/UPSTREAM.md`; complete baseline hashes in `UPSTREAM.json` | Recorded |
| Improve reliability | Runtime source-hash gate, rejection of unaudited source/config additions, rejection of old standalone contracts, import isolation, pinned-runtime lifecycle coverage | Implemented and tested |
| Cover the complete S4 sweep | All 16 window/lag pairs exercised with four layers, width 128, state size 64; finite forward/backward results | Passed on CPU |
| Reproducible model behavior and resume | Seed equality, causal-prefix check, exact optimizer/RNG continuation, checkpoint round trips, external numerical parity | Passed on CPU |
| Verify training rather than imports alone | All six external S4/S4ND synthetic fits, early stopping, checkpoint loading and evaluation now included in regular pytest | Passed on CPU |
| Side-by-side goals, contents and configurations | Generated `docs/S4_EXPERIMENTS.md`: 27 distinct configurations, exactly 125 canonical seeded runs; coverage and document freshness tested | Passed |
| GPU runtime readiness | Remote server reports no visible GPU | Not verified; requires an available GPU |

## Test environment

| Component | Version |
|---|---|
| Python | 3.10.10 |
| PyTorch | 2.9.1 |
| PyTorch Lightning | 2.0.4 |
| Hydra | 1.3.2 |
| OmegaConf | 2.3.0 |
| CUDA available locally | No |

The default global Python had Lightning 1.9.0. Its real external training smoke
failed because that version cannot construct the configured `strategy=auto`
trainer. Verification used an isolated system-site-packages overlay at
`/tmp/wctr-s4-checks.fhTpmo` with the project-pinned Lightning 2.0.4 installed;
the global installation was left unchanged. This is a CPU verification
environment, not a clean Linux/CUDA deployment lock.

Reproduction, after installing `requirements-benchmark.txt` in the chosen
environment:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q --disable-warnings -rs
python scripts/check_s4_upstream.py --upstream-checkout /path/to/authors-s4
python scripts/report_s4_experiments.py
python scripts/report_s4_experiments.py --format json
```

CPU thread limits above apply only to verification. Canonical experiment
hyperparameters, datasets, seed rosters and Transformer settings were not tuned
or changed by this work. Improved predictive accuracy has not been claimed;
that requires corrected experiment reruns.
