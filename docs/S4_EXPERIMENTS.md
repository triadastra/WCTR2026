# S4 experiment goals, contents and configurations

Generated from the current canonical launch contracts by `python scripts/report_s4_experiments.py`.
Use `--format json` for complete model, data, training, selection and source settings.

All rows use the actual modular `external/s4/src/models/sequence` implementation.
Forecasting/axial adapters and existing S4ND padding corrections are local integration; see `external/s4/UPSTREAM.md`.
Configuration inventory is not evidence of completed training or GPU qualification.

| Experiment / model | Direction | Goal | Contents | Layers / width / state | Window / lags | Batch | Seeds / runs |
|---|---|---|---|---|---|---|---|
| Exp 1 - s4 embeddings | causal | Aggregate forecasting and seed stability | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 33 | 25 |
| Exp 1 - s4 onehot | causal | Aggregate forecasting and seed stability | One-hot state/commodity/flow IDs + temporal S4 | 4 / 128 / 64 | 36 / 12 | 33 | 25 |
| Exp 1 - s4_bidir embeddings | bidirectional | Aggregate forecasting and seed stability | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 33 | 25 |
| Exp 1 - s4_bidir onehot | bidirectional | Aggregate forecasting and seed stability | One-hot state/commodity/flow IDs + temporal S4 | 4 / 128 / 64 | 36 / 12 | 33 | 25 |
| Exp 2 - s4 cross_attention_2d | causal | Compare categorical encoding and cross-series structure | State axial attention + commodity/flow embeddings (32/2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 1 | 3 |
| Exp 2 - s4 cross_attention_3d | causal | Compare categorical encoding and cross-series structure | State/commodity axial attention + flow embedding (2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 1 | 3 |
| Exp 2 - s4 cross_attention_4d | causal | Compare categorical encoding and cross-series structure | State/commodity/flow axial attention + temporal S4; no ID embeddings | 4 / 128 / 64 | 36 / 12 | 1 | 3 |
| Exp 2 - s4 embeddings | causal | Compare categorical encoding and cross-series structure | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 918 | 3 |
| Exp 2 - s4 onehot | causal | Compare categorical encoding and cross-series structure | One-hot state/commodity/flow IDs + temporal S4 | 4 / 128 / 64 | 36 / 12 | 918 | 3 |
| Exp 2 - s4_bidir cross_attention_2d | bidirectional | Compare categorical encoding and cross-series structure | State axial attention + commodity/flow embeddings (32/2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 1 | 3 |
| Exp 2 - s4_bidir cross_attention_3d | bidirectional | Compare categorical encoding and cross-series structure | State/commodity axial attention + flow embedding (2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 1 | 3 |
| Exp 2 - s4_bidir cross_attention_4d | bidirectional | Compare categorical encoding and cross-series structure | State/commodity/flow axial attention + temporal S4; no ID embeddings | 4 / 128 / 64 | 36 / 12 | 1 | 3 |
| Exp 2 - s4_bidir embeddings | bidirectional | Compare categorical encoding and cross-series structure | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 918 | 3 |
| Exp 2 - s4_bidir onehot | bidirectional | Compare categorical encoding and cross-series structure | One-hot state/commodity/flow IDs + temporal S4 | 4 / 128 / 64 | 36 / 12 | 918 | 3 |
| Exp 2 - s4nd 2d | bidirectional | Compare categorical encoding and cross-series structure | S4ND over state x commodity-flow-group x time; embeddings (32/2) | 4 / 128 / 64 | 36 / 12 | 1 | 3 |
| Exp 2 - s4nd 2d_state | bidirectional | Compare categorical encoding and cross-series structure | S4ND over state x time; commodity-flow groups in batch; embeddings (32/2) | 4 / 128 / 64 | 36 / 12 | 172 | 3 |
| Exp 2 - s4nd 3d | bidirectional | Compare categorical encoding and cross-series structure | S4ND over state x commodity x time; flow in batch + embedding (2) | 4 / 128 / 64 | 36 / 12 | 2 | 3 |
| Exp 2 - s4nd 4d | bidirectional | Compare categorical encoding and cross-series structure | S4ND over state x commodity x flow x time; no ID embeddings | 4 / 128 / 64 | 36 / 12 | 1 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 6 / 6 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 6 / 12 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 6 / 24 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 6 / 36 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 12 / 6 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 12 / 12 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 12 / 24 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 12 / 36 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 24 / 6 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 24 / 12 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 24 / 24 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 24 / 36 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 6 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 24 | 918 | 3 |
| Exp 3 - s4 embeddings | causal | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 36 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 6 / 6 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 6 / 12 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 6 / 24 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 6 / 36 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 12 / 6 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 12 / 12 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 12 / 24 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 12 / 36 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 24 / 6 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 24 / 12 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 24 / 24 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 24 / 36 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 6 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 12 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 24 | 918 | 3 |
| Exp 3 - s4_bidir embeddings | bidirectional | Measure sensitivity to history window and lag features | State/commodity/flow embeddings (8/32/2) + temporal S4 | 4 / 128 / 64 | 36 / 36 | 918 | 3 |

Total: **238 S4-family runs** across 50 distinct experiment/variant/window/lag configurations.

## Shared settings and interpretation

- All backbones: four layers, width 128, state size 64, `n_ssm=1`, dropout 0.1, pre-norm residual blocks and final LayerNorm. Width/depth matching does not imply equal parameter counts.
- 1-D S4: `s4` is causal, `s4_bidir` is bidirectional within the observed history. Both use the authors' temporal DPLR/FFT kernel, rank 1, HiPPO-LegS initialization, `dt_transform=softplus`, GELU/GLU and a last-time-step linear head. Directions have separate identities, output directories and checkpoints; parameter counts can differ.
- S4ND: bidirectional within the observed input grid, separable DPLR kernels, tensor rank 1; padded cells are excluded from layers and loss. No future target observations enter the input.
- `s4nd_2d_state` genuinely convolves over two axes. The historical `s4nd_2d` label actually uses three convolution axes (state, group, time); it is not identical to the state-only version.
- Axial attention: one single-head attention module per selected categorical axis, attention dropout 0.0, followed by four modular temporal S4 layers.
- Training: AdamW, fixed initial LR 0.0001 (no LR sweep), weight decay 0.0001 with author-marked SSM exemptions, betas (0.9, 0.999), epsilon 1e-8; no private kernel LR override. The existing plateau scheduler is retained.
- Selection: normalized MSE on value/weight, 5000-epoch safety cap, validation every epoch, early stopping patience 50 and minimum improvement 0.0001. Plateau scheduler patience 15, factor 0.5; gradient clipping 1.0; FP32, AMP/TF32 off.
- Exp 1 aggregates the eight-state panel to one series: batch 33, 132 training windows, four updates/epoch. Category IDs are constant in the aggregate arms.
- Exp 2 uses one full target month per update: 918 valid lanes, 132 updates/epoch. Flat batch 918, axial/grid batch 1, state-only S4ND batch 172 groups, S4ND 3D batch 2 flows are different layouts of that same month.
- Exp 3 uses one full target month per update, batch 918, common training start 2015-01, 108 updates/epoch. Every combination of windows/lags {6,12,24,36} is included.
- Train-only cohort and MinMax fit; train ends 2023-12, validation is 2024-01 through 2024-12, test is 2025-01 through 2025-05. All experiments predict the next month; window/lag values are not forecast horizons.
- `matched_axial_2d/3d/4d` are tested aliases of the corresponding `cross_attention_*` models, not extra paper runs.

Exp 1 seeds: 7, 13, 19, 26, 31, 44, 58, 67, 73, 88, 91, 104, 105, 115, 119, 127, 138, 152, 166, 173, 188, 194, 205, 211, 226.
Exp 2 seeds: 26, 105, 115.
Exp 3 seeds: 26, 105, 115.
