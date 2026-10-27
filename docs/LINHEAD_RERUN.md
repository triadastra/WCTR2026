# Linear-head Transformer/GPT rerun

## Why

In Experiments 1 and 2 every neural arm has width 128 and 4 blocks, but the
read-out differs:

| Arms | Head |
|---|---|
| LSTM, GRU, Mamba, Mamba-ND, S4, S4-bidir, S4 hydra, S4ND | `Linear(128 -> 2)` |
| Transformer (Exp 1/2) | `Linear(128 -> 64) -> ReLU -> Dropout -> Linear(64 -> 2)` (+8,128 params) |
| GPT (Exp 1/2) | `Linear(128 -> 128) -> ReLU -> Dropout -> Linear(128 -> 2)` (+16,512 params) |
| Transformer (Exp 3) | already `Linear(128 -> 2)` (`src/utils/exp3_protocol.py`) |

This rerun trains every Exp 1/2 Transformer and GPT arm again with the linear
head, so the head is equal across all neural arms. The published MLP-head runs
are kept unchanged; the rerun is an addition, not a replacement.

## What changes

* New model names `transformer_linhead` and `gpt_linhead`
  (`config/models/*_linhead.yaml`). They are the published architectures
  except `prediction_head: linear`. Both the flat and the axial (combo)
  classes honour the setting; the default stays `mlp`, so published
  checkpoints load and predict exactly as before (checked by re-running
  inference on four published checkpoints, 2026-10-16).
* `src/utils/linhead_protocol.py`: the roster is every published
  `transformer`/`gpt` run of Exp 1/2, renamed — 100 Exp 1 runs (2 models x
  onehot/embeddings x 25 seeds) and 36 Exp 2 runs (Transformer: onehot,
  embeddings, cross_attention, film_attention, cross_attention_{2,3,4}d;
  GPT: onehot, embeddings, cross_attention_{2,3,4}d; x seeds 26/105/115).
* `scripts/run_is_current.py --linhead` validates a run against that roster
  with the canonical contract (data, recipe, selection), so the usual
  completion and truth gates apply.
* Outputs go to `outputs/linhead/{agg,full}` (on the box:
  `/root/autodl-fs/wctr-linhead/runs`). The paper gate rejects any
  unexpected directory under `outputs/agg` or `outputs/full`, so the rerun
  never touches the published population or `results.xlsx`.

## Running on the H20 box

Nothing here starts on its own; the box is used only when the user says so.

1. Mac: bundle this branch and copy it with the data to the box.

   ```bash
   git bundle create /tmp/wctr-linhead.bundle claude/model-layer-differences-ea5274
   scp -i ~/.ssh/wctr_h20 -o IdentitiesOnly=yes -P <port> /tmp/wctr-linhead.bundle root@<host>:/root/autodl-tmp/
   ```

2. Box: create the release from the bundle; reuse the production runtime and
   the pinned data.

   ```bash
   mkdir -p /root/autodl-tmp/wctr-linhead && cd /root/autodl-tmp/wctr-linhead
   git clone -b claude/model-layer-differences-ea5274 /root/autodl-tmp/wctr-linhead.bundle repo
   cp -a /root/autodl-tmp/wctr-20260923-pr27/repo/data/processed/. repo/data/processed/
   bash repo/scripts/linhead_box.sh check
   ```

3. Train (resumable; rerun the same command after a container stop):

   ```bash
   bash repo/scripts/linhead_box.sh train
   bash repo/scripts/linhead_box.sh status
   ```

4. SHAP, after training finishes, one at a time:

   ```bash
   bash repo/scripts/linhead_box.sh shap1       # Exp 1, all 100 runs
   bash repo/scripts/linhead_box.sh shap2-flat  # Exp 2 1-D arms, seed 26, all lanes
   bash repo/scripts/linhead_box.sh shap2-nd    # Exp 2 axial arms, seed 26, all lanes
   bash repo/scripts/linhead_box.sh shap2-sum
   ```

   SHAP settings follow the published analyses: Exp 1 explains every run
   (`wctr-shap-exp1`); Exp 2 uses seed 26 and every kept lane
   (`analysis/exp2_shap_seed26`, script sha256 `4718f78a...`). The axial
   all-lanes SHAP is the expensive part (roughly 3 GPU hours per arm by the
   earlier estimate); it should follow whatever lane/model choice is made
   for the published ND arms.

   Unattended alternative: `scripts/linhead_after_train.sh` (copy to
   `/root/autodl-fs/wctr-linhead/after_train.sh`, run with nohup) resumes
   training if it is not running, waits for a clean finish, then runs Exp 1
   SHAP, Exp 2 flat SHAP and the two published Transformer flat arms that
   failed the 1e-4 gate on the Mac. The axial SHAP is left out on purpose.

5. Copy `/root/autodl-fs/wctr-linhead` to the external drive
   (`WCTR2026/box/autodl-fs/wctr-linhead`).

## Expected cost

Exp 1 took ~1.5 GPU min per run in production and Exp 2 ~6–14 min per flat
arm (axial arms longer): roughly 3 h for Exp 1 and 6–10 h for Exp 2 on one
H20, plus SHAP.

## Reporting

Report the linear-head arms next to the published ones (labels
"Transformer (linear head)", "GPT (linear head)"), not in place of them.
Head choice must not be selected on test metrics.
