# Review fixes — 9 September 2026

Scope: five reproduced findings from the repository, logs and experiment-design
review, plus related statistical publication bypasses discovered during repair.
Changes are local and preserve the original CSVs and historical result files.
The checkout already contained extensive unrelated changes; no commit or push
was made by this fix pass.

## Implemented

| Finding | Resolution | Regression evidence |
|---|---|---|
| Fabricated Exp 2/3 targets could pass metric checks | Reconstruct test outcomes from the pinned source; enforce exact series/date membership and both inverse transforms; calculate scaled metrics with reconstructed truth | Actual fabricated-target CLI reproduction now rejects; real-source mutation tests cover normalized truth, physical truth/predictions, identity and harmless row reordering |
| Statistical loader could relabel models/seeds or drop missing runs | Shared complete 156-run validator checks exact payload identity, recipe, completion, predictions and source truth before returning any runs | Missing/extra runs, mislabeled identities, failed completion/source checks, partial seeds and differing populations reject |
| Paired t test used normal tails | Student-t survival probability with n−1 degrees of freedom and explicit constant-difference behavior | At t=2 and n=8, p=0.0856193 rather than 0.0455003; reference tests match SciPy |
| Main, CD and axial tests used different units | All inferential paths use eight equally weighted state blocks after averaging months and lanes | Consumer tests capture eight-row matrices/differences; pooled-versus-state direction conflict handled explicitly |
| Friedman ties and alpha were wrong | Tie-corrected statistic; alpha-dependent studentized-range critical difference | Tied example returns statistic 16, not 12; all-equal and multiple-alpha cases checked |

The reporting audit additionally corrected unsafe cached MASE/paper-table
inputs, the `--plot-only` bypass, test-selected collapsing of S4ND layouts, and
stale workbook/figure captions. Both S4ND 2D layouts now have separate cells.
The paper generator consumes fresh validated Exp 1 rows and Exp 2 statistics;
it no longer expects obsolete significance columns in a cached workbook.
Automatic attachment of fresh statistics to an unchecked workbook was removed.

Statistical labels now describe within-CD non-rejection and exploratory
comparisons; they do not claim equivalence. Holm-adjusted, positive lane-weighted
comparisons supply paper-table daggers. BH results remain an explicitly
exploratory screening output with their assumptions stated. Numerical
references: [Student-t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html),
[Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html),
[BH](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

## Validation

- Real-source metric-gate tests: 14 passed.
- Shared-loader and existing integrity/provenance checks: 36 passed.
- Final focused numerical, reporting and loader suite: 45 passed. These suites
  overlap; their counts must not be added as a unique-test total.
- All six live statistical/paper entry points reject local legacy inputs. The
  original relabeling reproduction rejects. Hash comparison confirms existing
  workbooks and analysis files are unchanged by the rejected attempts.
- Synthetic figure execution completed for significance, CD and both baseline
  plots, including all-negative and zero improvements. These are rendering and
  control-flow checks, not experiment results.
- Full regression suite: **445 passed, 2 skipped, 47 warnings** in 1,039.70
  seconds. The two skips are CUDA/Triton-dependent Mamba round trips. Command:
  `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider`.
  OS access was enabled for PyTorch shared-memory workers and process cleanup.
- AST parsing passed for 90 first-party modules; `git diff --check` passed.
  `/tmp/wctr-fix-code-manifest.json` records 172 source/config/test hashes and
  installed runtime versions. No source/config/test changes occurred after
  that final snapshot; later modifications only finalized documentation.

Detailed execution logs are retained locally under `/tmp/wctr-fix-*`, with live
rejection checks in `/tmp/wctr-live-gates.log` and rendering diagnostics in
`/tmp/wctr-render-final.log`.

## Evidence limits

The software repairs do not produce a corrected empirical ranking. Historical
logs contain no matched fatal signatures but refer to the old cohort and lack
current completion evidence. CUDA/Triton qualification, complete canonical
reruns, source-ambiguity resolution, release-vintage assessment and a frozen
protocol on untouched forecast periods remain unverified. The source CSVs were
not replaced or silently deduplicated. See `RESULTS_STATUS.md` and `RERUN.md`.

Coordination uses `../message(astra-fable).md` (also linked from the repository).
Fable acknowledged the shared channel later on 2026-09-09: its previous notes
were in another clone, explaining the earlier lack of replies. Its worktree is
based on 5f1abee and this checkout on c5ce1b9 plus local changes; the test counts
refer to different code trees and are not interchangeable independent sign-off.

During reconciliation, the actual vendored S4 imports succeeded and 18
protocol/config tests passed. This checkout already uses two equal Exp 1
batches of 66 and overrides upstream S4 state size to the local value of 64.
Fable's corresponding protocol changes therefore must not be applied blindly.
The remaining stale SUMMARY.md, IMPLEMENTATION_STATUS.md and graphs/README.md
claims were corrected. No runtime/protocol changes or cross-clone merge were
made in this reconciliation. Fable received a reproduced zero-variance edge
case in its clustered test and feedback on bootstrap interpretation and
preserving the exact source/roster gates. Architecture/scan-coverage metadata
and a separately labeled lane-weighted sensitivity analysis remain candidates
for reviewed integration, not parts of this completed local fix pass.

## Integration update — 2026-09-13

The preceding test counts and clone status describe the September 9 snapshot.
That snapshot is preserved at `codex/review-snapshot-20260913` (2d6dece).
Integration now incorporates main at 6f55858, the 52-arm roster, architecture
metadata, S4 state size 64 and the author's lane-weighted primary objective.
Equal-state outputs are sensitivity analyses; see `STATISTICAL_PROTOCOL.md`.
Integration qualification is recorded separately after its tests finish.

The combined integration and follow-up validation are recorded in
[INTEGRATION_VALIDATION_2026-09-13.md](INTEGRATION_VALIDATION_2026-09-13.md).
