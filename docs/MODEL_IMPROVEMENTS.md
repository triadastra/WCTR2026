# Model changes — September 8, 2026

The generic `cross_attention` and `film_attention` encoders now use one token
each for State, Commodity and Flow. Separate projections bring their embeddings
to a common dimension; numeric features at each observed timestep supply the
queries. The FiLM encoder uses the resulting context to modulate numeric
features. Every parameter is created before optimizer setup.

This changes the architecture of per-series attention arms such as Transformer
CA/FiLM. Their old single-token checkpoints are incompatible with the new
parameter layout and must be evaluated with the original source revision.
Retrain into a new output directory; do not overwrite or relabel historical runs.

## New recurrent experiment

`identity_cross_attention` is supported by LSTM and GRU, in both training and
inference. It adds learned State/Commodity/Flow identity to the cross-group
queries and keys. Values remain numeric, and the existing raw-feature residual
and per-group recurrent head remain in place. It uses full group attention;
the existing latent and FiLM variants are separate experiments.

Example from the repository root:

```bash
python scripts/train.py --model lstm --variant identity_cross_attention --seed 42 --data-dir /path/to/data
python scripts/train.py --model gru --variant identity_cross_attention --seed 42 --data-dir /path/to/data
```

The new variant has its own configuration and run name. Default combo batch size
is one month, consistent with the existing recurrent combo variants.

## Corrections and logging

- LightGBM exposes `subsample_freq`, default 1, activating the configured 0.9
  row sampling. Set 0 explicitly to disable it. Resolved tree parameters are
  recorded in the run log.
- Masked attention returns zero for empty attention rows and zeroes padded
  combo outputs after normalization. A warning is recorded once per affected
  module, routed to the run logger during training.
- Shared sinusoidal positional encoding applies its dropout argument and
  supports odd feature widths. GPT's separate sinusoidal encoder also supports
  odd widths; its intentionally absent positional dropout remains unchanged.
- Bidirectional flat LSTM/GRU heads concatenate the final forward and backward
  hidden states. Unidirectional pooling remains the last timestep.
- Neural run logs record configuration, architecture versions, positional
  dropout, bidirectionality and attention guard behavior.

Mamba-ND and all vendored upstream source files were left unchanged by this work.
No benchmark was retrained, and these changes do not establish better forecasting
accuracy. Use the corrected data protocol for comparisons.

## Validation

155 selected CPU regression tests passed: model improvements, prior audit guards,
matched axial encoding, training logging, optimizer groups and training resume.
Checks cover nonzero query/key gradients, identity sensitivity, masked forward
and backward passes, checkpoint round-trip, actual LightGBM booster settings,
and run-file logging. The 22 edited/relevant Python files parsed successfully.

Validation used an isolated environment with CPU PyTorch 2.14.0, NumPy 1.26.4,
pandas 2.3.0 and LightGBM 4.6.0. This is not the full pinned GPU training
environment; Mamba/Triton execution and full benchmark retraining were not run.
