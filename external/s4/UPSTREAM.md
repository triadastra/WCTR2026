# S4 source provenance and benchmark adapters

Audited 2026-09-20 against the authors' official repository:
https://github.com/state-spaces/s4

Pinned upstream revision: `e757cef57d89e448c413de7325ed5601aceaac13`.
The comparison used a fresh checkout from that URL, not the README as proof.
`UPSTREAM.json` records upstream and local SHA-256 values for every compared
file. The baseline vendor tree had **423 byte-identical upstream files, 12
modified upstream files, 4 omitted upstream files, and 47 local additions**.
These audit documents themselves are not included in those baseline counts.
The directionality update promotes the existing bidirectional embeddings
recipe and adds a bidirectional one-hot recipe (48 local additions now).
Their hashes are recorded separately; none of the authors' core files changed.

## What is and is not rewritten

The modular **S4Block, FFTConv, DPLR/SSM kernels, HiPPO implementation, and
supporting Cauchy/Vandermonde/Krylov functions are byte-identical to upstream**.
They are actual authors' code, not a local reimplementation or an S4D substitute.

The **entire vendored repository is not pristine**. Existing changes are:

| Modified upstream file | Local adaptation |
|---|---|
| `models/s4/s4.py` | Standalone logging compatibility; no longer used by canonical benchmark arms |
| `src/dataloaders/__init__.py` | Trade-dataset registration |
| `src/dataloaders/base.py` | Dataset/loader integration |
| `src/dataloaders/et.py` | Dataset compatibility |
| `src/dataloaders/ts.py` | Trade tensor/grid builders |
| `src/models/nn/normalization.py` | Guard constant padded grid cells against division by zero |
| `src/models/sequence/backbones/block.py` | Apply grid masks around residual blocks |
| `src/models/sequence/backbones/model.py` | Propagate and apply optional grid masks |
| `src/models/sequence/modules/s4nd.py` | Mask padded inputs/outputs; the S4ND file is therefore not byte-identical |
| `src/tasks/encoders.py` | Trade identity encoders |
| `src/utils/train.py` | Allow repeated OmegaConf resolver registration |
| `train.py` | Benchmark training, masked losses, checkpoint provenance/completion, local CSV logging and checkpoint inventory integration |

Local additions include trade adapters, experiment configurations and a grid
mask helper. These are necessary benchmark integration, not evidence that the
whole repository is untouched. Existing correctness fixes are retained.

The 2026-09-22 logging update also imports the benchmark-side
`src/utils/s4_logging.py` adapter for local epoch records, heartbeats and failure
events. `UPSTREAM.json` records the updated training-harness hash; the ten
designated authors' S4 core files remain unchanged.

## One modular S4 backbone across the benchmark

`src/models/s4.py` is now only a forecasting/grid adapter around the same
`SequenceModel` used by the external runner. `src/models/s4_modular.py` imports
the vendor files without editing, copying, or translating their source. It
temporarily scopes the authors' `src` namespace during construction, then
restores the benchmark namespace. Forward/backward run the actual vendor code.
Use state-dict checkpoints, as the benchmark already does; full-object Python
pickle is not supported across the colliding package namespaces.

| Experiment path | Implementation |
|---|---|
| Exp 1 aggregate S4, embeddings and one-hot | Modular SequenceModel through forecasting adapter |
| Exp 2 flat S4, embeddings and one-hot | Existing modular Lightning runner, unchanged |
| Exp 2 S4 axial 2D/3D/4D and aliases | Local axial encoding followed by modular SequenceModel |
| Exp 3 every S4 window/lag cell | Modular SequenceModel through forecasting adapter |
| S4ND layouts | Existing modular S4ND and documented padding adapters, unchanged |

The switched S4 arms explicitly use `n_ssm=1`, `dt_transform=softplus`, tied
residual dropout, pre-normalized residual blocks and the authors' final LayerNorm,
matching the existing Exp 2 flat-S4 recipe. The shared benchmark LR is retained
by clearing the inherited private kernel LR. Width 128, four layers, state size
64, data, seeds, training and evaluation policies are otherwise unchanged.
Transformer/GPT feed-forward widths have not been changed by this migration.

Contracts for the switched S4 arms record `implementation: authors-modular-s4-v1`;
old standalone contracts and state dictionaries must not be resumed or reused under this
identity. Old output artifacts are preserved, not relabeled. GPU qualification
and corrected full experiment runs remain separate from CPU regression tests.

## Verification

Offline verification of the audited local files:

```sh
python scripts/check_s4_upstream.py
```

Independent comparison against a checkout at the pinned upstream commit:

```sh
python scripts/check_s4_upstream.py --upstream-checkout /path/to/authors-s4
```

Both commands fail if any audited bytes differ. The second also checks the
upstream commit and all reference bytes. Audit hashes establish source identity,
not predictive quality or full GPU/runtime readiness.
Added Python/YAML files under `src`, `models`, or `configs` are also rejected
until explicitly audited. The benchmark runtime preflight invokes this check,
so a successful import alone cannot conceal a changed vendor tree.

The migration regression tests cover all eight native S4 variant names,
forward/backward, optimizer steps, state-dict reloads, namespace restoration,
and rejection of old standalone contracts. A fixed-state CPU comparison also
matched outputs, input gradients and all parameter gradients exactly against
the fresh, unmodified upstream checkout at the pinned revision. This is a
numerical check on that fixture, not a claim of GPU or full-training equivalence.
Additional tests exercise every canonical S4 window/lag shape with the full
four-layer, width-128, state-64 backbone, causal prefixes, seed reproducibility,
and exact optimizer/RNG resume. `docs/S4_EXPERIMENTS.md` is generated from the
launch contracts and checked for complete roster coverage and staleness.
The regular pytest suite also runs the six external S4/S4ND synthetic training,
early-stopping, checkpoint and evaluation lifecycles, using the pinned benchmark
dependencies. Lightning 1.9 is not compatible with the configured trainer;
`requirements-benchmark.txt` pins Lightning 2.0.4.
