# Model Differences

This file records differences found while verifying `code/github/src/models/*`
against the paper-producing implementations in `code/paper included/`.

## GRU Verification — Task 1.2

Files compared:

- New: `src/models/gru.py`
- Old:
  - `code/paper included/GRU/model.py`
  - `code/paper included/GRU/GRU_flat.py`
  - `code/paper included/GRU/GRU_onehot.py`
  - `code/paper included/GRU/GRU_combo.py`
  - `code/paper included/GRU/GRU_combo_caonly.py`

### Construction Checks

The current `GRUModel` constructs and forwards for all four declared variants:

| Variant | New output shape | New params | New input dim |
|---|---:|---:|---:|
| `onehot` | `(3, 2)` | `407362` | `135` |
| `embeddings` | `(3, 2)` | `386346` | `72` |
| `cross_attention` | `(3, 2)` | `408874` | `72` |
| `film_attention` | `(3, 2)` | `419554` | `72` |

Old reference class shape checks:

| Old class | Input shape | Output shape | Params |
|---|---|---:|---:|
| `GRU` | `(B, L, F)` | `(B, 2)` | `405634` with `F=70` |
| `GRUWithCrossAttention` | `(B, L, G, F)` | `(B, G, 2)` | `494595` with `G=5, F=28` |
| `GRUWithCrossAttentionOnly` | `(B, L, G, F)` | `(B, G, 2)` | `411522` with `G=5, F=28` |

### Differences

The new `GRUModel` is shape-compatible for the public training path, but it is
not architecturally equivalent to the paper-producing GRU implementations.

1. **Base GRU block differs.**
   - Old `model.py::GRU` uses `input_proj: Linear(input_dim, hidden_size)`,
     then `nn.GRU(input_size=hidden_size, ...)`, then
     `output_proj: Linear(hidden_size, output_dim)`.
   - New `src/models/gru.py::GRUModel` feeds encoded features directly into
     `nn.GRU(input_size=self.input_dim, ...)`, then applies a two-layer output
     MLP: `Linear(H, H/2) -> ReLU -> Dropout -> Linear(H/2, 2)`.
   - This means state dicts are not directly compatible and parameter counts
     differ even when tensor shapes match.

2. **Embedding flow dimension differs from the old embedding variant.**
   - Old `GRU_flat.py::Cfg.flow_emb_dim` is `2`.
   - New `config/base.yaml` and `config/variants/embeddings.yaml` use
     `embedding_dims.flow: 4`, giving effective input dimension `72` instead
     of the old `70` for `F_numeric=28`.

3. **Cross-attention variants are semantically different.**
   - Old `GRU_combo.py` / `GRU_combo_caonly.py` use combo-window tensors
     shaped `(B, L, G, F)` and perform attention across all
     `(State, Commodity, Import/Export)` groups per timestep.
   - Old CA + FiLM outputs per-group predictions `(B, G, 2)`, with masked
     losses over valid groups.
   - New `variant="cross_attention"` and `variant="film_attention"` encode
     only the categorical IDs for each individual series, concatenate the
     result to `(B, L, F)`, and output one next-step prediction `(B, 2)`.
   - Therefore the new CA/FiLM variants do not reproduce the old paper
     CA-only or CA+FiLM GRU experiments.

### Resolution (2026-05-23)

**Base block ported.** `src/models/gru.py::GRUModel` now matches
`paper included/GRU/model.py::GRU` for `onehot` and `embeddings`:

- `input_proj: Linear(input_dim -> hidden_size)`
- `gru: nn.GRU(hidden_size, hidden_size, num_layers, dropout)`
- `output_proj: Linear(hidden_size -> 2)` (single Linear, not the 2-layer MLP)
- Prediction taken from `gru_out[:, -1, :]` (last timestep of the full
  output sequence). For a unidirectional single-batch GRU this equals
  `h_n[-1]`, but staying with the paper's expression preserves state-dict
  key parity.

**`flow_embed_dim` aligned.** `config/base.yaml` and
`config/variants/embeddings.yaml` both now set `embedding_dims.flow: 2`
to match `paper included/GRU/GRU_flat.py::Cfg.flow_emb_dim = 2`. Total
categorical embedding size for the embeddings variant is now
`8 + 32 + 2 = 42` (was `44`).

**CA / FiLM variants — still divergent. Tracked as Phase 1.5.**
The current `cross_attention` and `film_attention` variants in
`src/models/gru.py` are runnable simplified per-series encoders. They are
NOT equivalent to the paper's CA-only and CA+FiLM GRU experiments, which
use a `(B, L, G, F)` combo-window pipeline with group masks and per-group
losses. The same issue affects `src/models/lstm.py` (paper-included has
`LSTM_combo.py` / `LSTM_combo_caonly.py` with the identical pattern).

Proper paper-equivalent CA / FiLM requires:

1. A new dataset class that emits `(B, L, G, F)` combo-window tensors
   (mirror `paper included/shared_dataloader.py::ComboWindowDataset`).
2. Per-group masks for valid `(state, commodity, flow)` combinations.
3. Per-group losses (mask-aware MSE/MAE) in `src/training/losses.py`.
4. Model variants that consume `(B, L, G, F)` and output `(B, G, 2)`.

That work is sizable (multiple files, affects trainer, affects metrics)
and applies to LSTM and GRU identically. Promoted to its own phase: see
plan.md **Phase 1.5 — Combo-window data path**. Do not block Phase 1
closure on it; the base variants are what the equivalence gate validates
first.

---

## Transformer Verification — Task 1.3

Files compared:

- New: `src/models/transformer.py`
- Old:
  - `code/paper included/Transformer/flat.py`
  - `code/paper included/Transformer/onehot.py`

### Construction Checks

The `TransformerModel` constructs and forwards for all four declared variants.
Output shape is `(B, 2)` on random input of shape `(B, L, F_in)`.

| Aspect | Old `FlatTransformer` | New `TransformerModel` | Verdict |
|---|---|---|---|
| Architecture style | Encoder-only | Encoder-only | ✅ Match |
| Positional encoding | Sinusoidal, `max_len=1024` | Sinusoidal, `max_len=5000` | ✅ Equivalent |
| Encoder layers | `nn.TransformerEncoderLayer` × 4 | `nn.TransformerEncoderLayer` × N | ✅ Match |
| Attention heads | `nhead=8` | `num_heads=8` (configurable) | ✅ Match |
| Pooling | Last token `h[:, -1]` | Last token `x[:, -1, :]` | ✅ Match |
| Output head | `Linear→ReLU→Dropout→Linear` | `Linear→ReLU→Dropout→Linear` | ✅ Match |
| Pre-head norm | `nn.LayerNorm(d_model)` on last token | **None** | ⚠️ Difference |
| Default `d_model` | 256 | 128 (configurable) | Configurable |
| Default `dim_feedforward` | 768 (3× d_model) | 512 (4× d_model) | Configurable |
| Activation | Default `relu` | Explicit `gelu` | Configurable |
| Encoding | Inline embeddings / one-hot | `EncodingFactory` | By design |

### Differences

1. **Missing pre-head LayerNorm.** The old code applies `nn.LayerNorm(d_model)` to the
   last token before the output MLP. The new code does not. This is a minor
   architectural difference; both paths produce `(B, 2)` output.

2. **Default hyperparameters differ.** Old defaults (`d_model=256`, `dim_feedforward=768`,
   `dropout=0.1`) do not match new config defaults (`hidden_size=128`, `d_ff=512`).
   The new `config/models/transformer.yaml` should be updated to match paper defaults
   if bit-exact reproduction is required.

3. **Encoding delegation.** The old code builds embeddings inline in the training
   script; the new code delegates to `EncodingFactory`. This is the intended
   unified design and does not affect transformer core architecture.

### Recommendation

The `TransformerModel` is architecturally equivalent to the paper's `FlatTransformer`
(encoder-only, sinusoidal PE, last-token pooling, MLP head). The missing LayerNorm
and default hyperparameter mismatch are the only differences. Update
`config/models/transformer.yaml` to match paper defaults for reproduction gates.

---

## XGBoost Verification / Fix — Task 1.3

Files compared:

- New: `src/models/xgboost.py`
- Old:
  - `code/paper included/xgboost/flat.py`
  - `code/paper included/xgboost/onehot.py`
  - `code/paper included/xgboost/xgboost_combo.py`

### Issue Found (P0 — blocks training)

`XGBoostModel.__init__` did **not** call `super().__init__()`, so:
- `nn.Module` was never initialized → `.to()`, `.parameters()`, `get_num_params()` failed.
- `BaseModel.__init__` was skipped → `_build_model()` never ran, `self.xgb_params` unset.
- `Trainer.fit()` crashed on `model.to(device)` and `model.parameters()`.

### Fix Applied

1. **`src/models/xgboost.py`** — `XGBoostModel.__init__` now calls
   `super().__init__(...)` so `BaseModel` creates the encoder and invokes
   `_build_model()`. Added `self.requires_training = False` to signal the Trainer.
   Added `_extract_numpy(loader)` helper that runs `encode_features()` on each
   batch, flattens the time dimension `(B, L, F) → (B, L*F)`, and returns stacked
   numpy arrays. Updated `fit()` and `predict_loader()` to accept PyTorch
   DataLoaders instead of raw numpy.

2. **`src/training/trainer.py`** — `Trainer.fit()` now checks
   `getattr(model, 'requires_training', True)` at entry. If `False`, it branches
   to `_fit_sklearn()`, which:
   - Skips optimizer / scheduler / early-stopping setup.
   - Calls `model.fit(train_loader, val_loader)`.
   - Validates via `model.predict_loader(val_loader)` and computes MSE.
   - Saves checkpoint via `model.save()`.
   - Returns a history dict compatible with the PyTorch path.

### Resolution (2026-05-24)

**Paper-style supervised rows ported.** `src/models/xgboost.py` no longer
flattens the full 36-month sequence. It now mirrors
`paper included/xgboost/flat.py` and `onehot.py`:

- Use one tabular row per adjacent `(series, timestep)` pair after lag burn-in.
- Rebuild rows directly from the unified `TradeDataset`'s normalized panel
  instead of inheriting the RNN 36-window sample restriction.
- Drop current `Value` / `Weight` from the feature row to avoid leakage.
- Keep `value_lag_1..12`, `weight_lag_1..12`, `sin_month`, `cos_month`.
- Add categorical features:
  - `variant="embeddings"` means integer ID columns (`state_id`, `comm_id`,
    `flow_id`), matching the old `flat.py` naming despite not using learned
    neural embeddings.
  - `variant="onehot"` means explicit one-hot state / commodity / flow columns,
    matching old `onehot.py`.
- Train separate `XGBRegressor` models for value and weight.
- Align default XGBoost hyperparameters with the paper scripts:
  `n_estimators=2000`, `learning_rate=0.05`, `max_depth=8`,
  `subsample=0.9`, `colsample_bytree=0.8`, `tree_method="hist"`,
  `random_state=42`.

**No longer a difference (corrected 2026-08-24).** This previously read that
the unified pipeline "uses the Phase 0 corrected burn-in behavior (drops
2008-2011 for 36 window + 12 lags)" against the old loader's drop-2008. That
Phase 0 behaviour was a defect, not a correction: it deleted the `input_len`
months of history the first target consumes, so the window requirement was
applied twice and 36 target months were lost per series. The burn-in now drops
`lag_count` months at month granularity, which is the same contract the old
loader satisfied. See `docs/BEHAVIOR_DIFFERENCES.md` and `debug.md` R33.

**A real difference remains here, on the tabular side.** `extract_from_trade_dataset`
builds one row per adjacent `(series, timestep)` pair, so the tree arms need
only 1 row of history where the neural arms need `input_len`. They therefore
start `input_len - 1` months earlier and train on more targets — currently
32,130 more samples, invariant to the burn-in fix. Tracked as R13 in
`debug.md`; not yet resolved.

**Verification still needed on VM.** Local checks can validate imports and row
construction, but the full XGBoost train remains a remote-GPU/VM gate by project
policy.

---

## LightGBM — Task 1.6

LightGBM is a new public-repo benchmark, not a paper reproduction target. The
paper did not report LightGBM numbers.

### Architecture / Data Path

`src/models/lightgbm.py::LightGBMModel` is a sklearn-style tabular
gradient-boosting baseline. It uses the same supervised-row construction as the
paper-equivalent XGBoost path:

- one row per adjacent `(series, timestep)` pair after lag burn-in;
- features from `src/models/tabular_common.py`;
- `Value` / `Weight` excluded from features to avoid leakage;
- lag columns, seasonal columns, and categorical identity columns retained;
- separate regressors for value and weight.

Both `XGBoostModel` and `LightGBMModel` now import their row construction from
`src/models/tabular_common.py`, so XGB-vs-LightGBM comparisons differ by the
boosting implementation rather than feature assembly.

### Why Included

XGBoost was the only gradient-boosting baseline in the original refactor.
LightGBM is the obvious peer within the same family and is often faster on
large tabular data. Adding it tests whether the qualitative finding that
boosted trees are strong on this dataset survives a change in boosting library.

### Hyperparameters

Defaults are LightGBM analogues of the XGBoost paper settings:

- `n_estimators=2000`
- `learning_rate=0.05`
- `num_leaves=255`
- `max_depth=-1`
- `subsample=0.9`
- `colsample_bytree=0.8`
- `random_state=42`

`num_leaves` is the primary LightGBM tree-shape control; `max_depth=-1` leaves
depth unrestricted while constraining tree complexity through leaf count.
