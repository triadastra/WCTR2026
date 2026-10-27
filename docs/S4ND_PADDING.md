# S4ND occupancy and dimensionality contract

Updated 16 September 2026. Corrected grid models require retraining; no corrected
production results are claimed by this change.

## Missing cells

Occupancy comes from the frozen training cohort, never from the target's value.
A real zero-valued observation remains valid. Structural holes retain their
categorical coordinates and do not become additional samples.

All four S4ND grid arms share this behavior:

1. Zero absent numeric inputs **before** encoding, including NaN/Inf padding.
2. Zero encoded cells again, removing embedding and projection offsets there.
3. Mask before each convolution, after normalization, after each residual block,
   and after the final normalization. Mask the S4ND layer's own input/output too.
   Thus absent cells cannot store intermediate signals and feed them into the
   next layer. Transposed scalar normalization guards zero variance.
4. Zero absent readouts and select valid prediction/target pairs before computing
   loss and metrics. MSE is summed over valid cells and both outputs, then divided
   by `valid_cells * 2`, exactly once. Epoch logging uses valid-cell counts.

The earlier all-cells/valid-cells loss multiplier was redundant after target
selection and inflated gradients. It is removed. Training and evaluation use
the same forward path and occupancy resolver. Invalid/missing occupancy metadata
fails explicitly instead of silently admitting padded cells.

This is fixed-lattice masked convolution: valid cells can interact across a hole
through the convolution kernel, but the hole is not an active token. Kernel
distances retain the original categorical indexing. There is no occupancy-based
kernel renormalization or compaction, and no claim that this is mathematically
identical to masked attention or Mamba-ND's compacted scans.

## Dimensionality ladder

Let `B` be the number of samples, `L` the history length, and `F` the input
feature count. The trailing prediction dimension is Value/Weight (`2`).

| Arm | Numeric input | Convolved axes | Axes placed in the batch |
|---|---|---|---|
| `s4nd_2d_state` | `[B, State, L, F]` | State × Time | Commodity–Flow groups |
| `s4nd_3d` | `[B, State, Commodity, L, F]` | State × Commodity × Time | Flow |
| `s4nd_4d` | `[B, State, Commodity, Flow, L, F]` | State × Commodity × Flow × Time | Target month |

These are the 2D → 3D → 4D members of the axis-expansion comparison. The separate
`s4nd_2d` arm has input `[B, State, Group, L, F]`, where Group combines Commodity
and Flow. It actually convolves over **three axes**, State × Group × Time, and
must be reported as a folded-group variant rather than the genuine 2D rung.
All arms preserve the common valid series–target-month observations. Equal
width/depth does not imply equal parameter count or compute.

## Checkpoint boundary and verification

Grid run contracts now record
`grid_padding=fixed-lattice-masked-input-and-layers-v1` and
`loss_reduction=valid-target-mean-v2`. Resume, evaluation, result reuse and
collection compare these against the current expected contract. A checkpoint
without these semantics is incompatible; changing its metadata cannot repair
its learned weights. The vendored contract also reads the shared benchmark
protocol version instead of carrying a stale copy.

`tests/test_s4_padding.py` runs the real encoders, three-layer S4ND backbones,
decoder, training loss and checkpoint hooks for all four grids in both layouts.
It checks exact prediction invariance to finite/NaN/Inf padding, zero padded
input gradients, zero inter-layer padded activations, finite parameter gradients,
agreement with a manual valid-observation loss and gradients, unchanged behavior
with an all-valid mask, and rejection of obsolete checkpoint/result contracts.

`tests/smoke_exp2_s4.py` exercises all six S4/S4ND arms through CPU Lightning
training, early stopping, selected-checkpoint reload, completion-receipt
validation and evaluation. Its zero learning rate makes stopping deterministic;
this validates lifecycle wiring, not convergence. CUDA kernels, production-size
memory use and corrected benchmark accuracy need validation on the training host.

Local verification on 16 September: the focused regression group passed 115
tests; the subsequently expanded padding/contract file passed all 17 tests
(overlapping the regression group). All six CPU lifecycle smoke runs passed.
