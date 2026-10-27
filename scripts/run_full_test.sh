#!/usr/bin/env bash
# Full benchmark run for the WCTR paper rewrite (the "full test"):
# every model x variant on the train-only sparse-filtered 918-combo set,
# window 36 / lag 12 / 4 layers / 1-step walk-forward.
#
# All artifacts land under one folder with standardized names:
#   $RUNS/<model>_<variant>_seed<seed>/   (checkpoint.pt, metrics.json, predictions.csv)
# S4-family runs (external/s4) are copied into the same convention as
#   $RUNS/s4_<experiment>_seed<seed>/
#
# Resumable at two levels:
#   * between runs -- a run is skipped only when metrics.json proves it used the
#     corrected train-only cohort. Legacy 903-combo artifacts are rerun.
#   * within a run -- RESUME=1 continues a run that was killed mid-training from
#     its last.pth (optimizer, LR schedule and early-stopping counter included)
#     instead of restarting it at epoch 0. Off by default so a plain invocation
#     stays byte-identical to the published protocol.
# Usage: bash scripts/run_full_test.sh [seed]              (default seed 42)
#        RESUME=1 bash scripts/run_full_test.sh [seed]     (continue killed runs)

set -uo pipefail
cd "$(dirname "$0")/.."

SEED="${1:-42}"
RUNS="outputs/full"
if [ "${1:-}" = "--print-plan" ]; then
    exec "${PY:-${PYTHON:-python}}" scripts/prepare_exp2.py
fi
DATA_DIR="${DATA_DIR:-.}"       # canonical CSV: $DATA_DIR/data/processed/
RESUME="${RESUME:-0}"           # 1 = continue killed runs from last.pth
PY="${PY:-${PYTHON:-python}}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
command -v "$PY" >/dev/null 2>&1 || {
    echo "[preflight FAIL] Python executable not found: $PY" >&2
    exit 2
}
PY="$(command -v "$PY")"
PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"
if [ ! -d "$DATA_DIR" ]; then
    echo "[preflight FAIL] data root does not exist: $DATA_DIR" >&2
    exit 2
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
export DATA_DIR
if ! "$PY" scripts/check_legacy_data.py --data-dir "$DATA_DIR" >/dev/null; then
    echo "[preflight FAIL] configured source fingerprint/domain/duplicate audit failed" >&2
    exit 2
fi
if ! "$PY" scripts/prepare_exp2.py --check-data --check-runtime --data-dir "$DATA_DIR"; then
    echo "[preflight FAIL] Experiment 2 runtime/recipe validation failed" >&2
    exit 2
fi
# Membership is sourced only after the data-only manifest parser validates it.
"$PY" -m src.utils.experiment_manifest --repo . >/dev/null || exit 2
source config/experiment_manifest.env
mkdir -p "$RUNS"
LOCK_DIR="$RUNS/.seed${SEED}.lock"
mkdir "$LOCK_DIR" 2>/dev/null || { echo "[FAIL] seed $SEED already has a controller" >&2; exit 2; }
trap 'rmdir "$LOCK_DIR"' EXIT
FAILED=0

# Under RESUME the previous attempt's log must survive: `tee` (no -a) truncates
# it, which would throw away the very epochs the resume is continuing from.
RESUME_FLAG=""; TEE_FLAG=""
[ "$RESUME" = "1" ] && { RESUME_FLAG="--resume"; TEE_FLAG="-a"; }

is_current () {
    local metrics="$1"; shift
    "$PY" scripts/run_is_current.py "$metrics" --exp2 --input-len 36 --lag-count 12 --train-target-start null --expected-seed "$SEED" "$@"
}

run_github () {  # run_github <model> <variant>
    local model="$1" variant="$2"
    local dir="$RUNS/${model}_${variant}_seed${SEED}"
    if is_current "$dir/metrics.json" --expected-model "$model" --expected-variant "$variant"; then
        echo "[skip] $dir (current corrected metrics)"; return 0
    fi
    mkdir -p "$dir"
    rm -f "$dir/metrics.json"
    local run_resume="$RESUME_FLAG"
    if [ "$RESUME" = "1" ] && [ -f "$dir/last.pth" ] && ! \
        "$PY" scripts/prepare_exp2.py --native-checkpoint "$dir/last.pth" \
            --model "$model" --variant "$variant" --seed "$SEED"; then
        run_resume=""
        echo "[fresh] $model/$variant: prior checkpoint is off-protocol"
    fi
    echo "[train] $model/$variant -> $dir"
    if ! "$PY" scripts/train.py --model "$model" --variant "$variant" \
        --seed "$SEED" --data-dir "$DATA_DIR" --out-dir "$dir" \
        $run_resume 2>&1 | tee $TEE_FLAG "$dir/train.log"; then
        echo "[FAIL] training $model/$variant failed; inference suppressed"
        return 1
    fi
    local ckpt="$dir/best.pth"
    case "$model" in
        # trainer.py saves both tree models under the "best_xgb" prefix;
        # their .load() takes the prefix and appends _value/_weight.joblib.
        xgboost|lightgbm|random_forest) ckpt="$dir/best_xgb" ;;
    esac
    echo "[infer] $model/$variant ($ckpt)"
    if ! "$PY" scripts/inference.py --model "$model" --variant "$variant" \
        --checkpoint "$ckpt" --seed "$SEED" --device cuda \
        --data-dir "$DATA_DIR" --out-dir "$dir" --normalized \
        2>&1 | tee "$dir/inference.log"; then
        return 1
    fi
    is_current "$dir/metrics.json" --expected-model "$model" --expected-variant "$variant"
}

run_baseline () {  # run_baseline <name>
    local name="$1"
    local dir="$RUNS/${name}_none_seed${SEED}"
    if is_current "$dir/metrics.json" --expected-model "$name" --expected-variant none; then
        echo "[skip] $dir"; return 0
    fi
    mkdir -p "$dir"
    rm -f "$dir/metrics.json"
    # --seed matters for the LABEL, not the math (baselines are deterministic):
    # without it every ${name}_none_seed${SEED} dir records payload seed 42,
    # which build_paper_results.py trusts over the dir name — the workbook then
    # shows three identical "seed 42" rows and n_seeds=1 for baselines.
    if ! "$PY" scripts/run_baseline.py --model "$name" --seed "$SEED" \
        --data-dir "$DATA_DIR" --out-dir "$dir" --normalized 2>&1 | tee "$dir/run.log"; then
        return 1
    fi
    is_current "$dir/metrics.json" --expected-model "$name" --expected-variant none
}

s4_identity () {  # s4_identity <experiment> -> "model variant"
    case "$1" in
        s4_embeddings) echo "s4 embeddings" ;;
        s4_onehot) echo "s4 onehot" ;;
        s4_embeddings_bidir) echo "s4_bidir embeddings" ;;
        s4_onehot_bidir) echo "s4_bidir onehot" ;;
        s4nd_2d) echo "s4nd 2d" ;;
        s4nd_2d_state) echo "s4nd 2d_state" ;;
        s4nd_3d) echo "s4nd 3d" ;;
        s4nd_4d) echo "s4nd 4d" ;;
        *) return 1 ;;
    esac
}

run_s4 () {  # run_s4 <experiment> [extra hydra overrides...]
    local exp="$1"; shift
    local dir="$RUNS/s4_${exp}_seed${SEED}"
    local identity model variant
    identity="$(s4_identity "$exp")" || {
        echo "[FAIL] unknown canonical S4 experiment: $exp" >&2
        return 2
    }
    read -r model variant <<< "$identity"
    if is_current "$dir/metrics.json" --expected-model "$model" --expected-variant "$variant"; then
        echo "[skip] $dir (current corrected metrics)"; return 0
    fi
    mkdir -p "$dir"
    rm -f "$dir/metrics.json"
    local best="$dir/hydra/checkpoints/val/loss.ckpt"
    local last="$dir/hydra/checkpoints/last.ckpt"
    # A marker alone does not establish the checkpoint's recipe. Evaluate it
    # through the strict contract gate; otherwise retrain under current settings.
    if [ -f "$dir/training_complete" ] && [ -f "$best" ]; then
        if (cd external/s4 && "$PY" ../../scripts/s4_eval.py "$exp" --seed "$SEED") && \
            is_current "$dir/metrics.json" --expected-model "$model" --expected-variant "$variant"; then
            return 0
        fi
        rm -f "$dir/training_complete" "$dir/metrics.json"
    fi
    local resume_override=""
    if [ "$RESUME" = "1" ] && [ -f "$last" ] && \
        "$PY" scripts/prepare_exp2.py --s4-checkpoint "$last" --experiment "$exp" --seed "$SEED"; then
        resume_override="train.ckpt=$(cd "$(dirname "$last")" && pwd)/$(basename "$last")"
        echo "[resume] $exp from $last"
    else
        # A deliberate fresh run may replace an incomplete Hydra directory.
        rm -rf "$dir/hydra"
    fi
    echo "[train] external/s4 experiment=$exp -> $dir"
    ( cd external/s4 && WANDB_DISABLED=true \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      "$PY" train.py \
        "experiment=$exp" "train.seed=$SEED" "$@" \
        "dataset.data_dir=$DATA_DIR" \
        "wandb=null" "loader.num_workers=0" \
        "hydra.run.dir=../../$dir/hydra" ${resume_override:+"$resume_override"} ) \
        2>&1 | tee $TEE_FLAG "$dir/train.log"
    # PIPESTATUS[0], not the pipeline's status: with `... | tee` the exit code
    # belongs to tee, which succeeds even when training crashed or was killed,
    # so `&& touch done` marked interrupted runs complete and the resumable
    # skip then passed over them forever. One arm was recorded done after being
    # killed at epoch 12 of a run that should have gone much further.
    local rc="${PIPESTATUS[0]}"
    if [ "$rc" -eq 0 ]; then
        touch "$dir/training_complete"
        if ! (cd external/s4 && "$PY" ../../scripts/s4_eval.py "$exp" --seed "$SEED"); then
            return 1
        fi
        is_current "$dir/metrics.json" --expected-model "$model" --expected-variant "$variant" || return 1
    else
        echo "[FAIL] $exp exited $rc — no done marker written, will retry on the next pass"
    fi
    return "$rc"
}

# ---- Tier 1: per-series models, both basic encoders -------------------------
# mamba = real Mamba-2 (vendored state-spaces/mamba); GPU/triton only, trained here
# with the rest of the capacity-matched (d_model=128) per-series set.
# NOTE: "dss" is NOT here — the real Diagonal State Space runs in Tier 4 via the
# upstream s4 repo (complex-diagonal kernel); the src/models/dss.py gated-RNN stub
# is retired from the benchmark.
for model in $EXP2_BASIC_MODELS; do
    for variant in $BASIC_VARIANTS; do
        run_github "$model" "$variant" || FAILED=1
    done
done

# ---- Tier 2: combo (cross-attention) variants --------------------------------
for model in $EXP2_COMBO_MODELS; do
    for variant in cross_attention film_attention; do
        run_github "$model" "$variant" || FAILED=1
    done
done

# ---- Tier 2a: transformer + flat cross/FiLM encoders -------------------------
# NOT the same mechanism as Tier 2 despite sharing the variant names. The
# `combo` predicate in scripts/train.py admits transformer only for the AXIAL
# variants, so these two land on the per-series path and get
# encodings.CrossAttentionEncoding / FiLMAttentionEncoding: numeric-timestep
# queries attending over THREE per-category key tokens (State, Commodity,
# Flow), architecture_version categorical_attention_v2 (commit 5f1abee,
# docs/MODEL_IMPROVEMENTS.md). Until 2026-09-08 that encoder had exactly one
# key -- softmax identically 1.0, W_q/W_k with zero gradient forever, a fixed
# linear map -- and the recorded transformer_cross_attention /
# transformer_film_attention rows in results.xlsx (847k / 836k params,
# film ranked SECOND) were produced by THAT degenerate encoder. The rerun
# trains a different model under the same arm name; do not compare the two
# (RERUN.md). Per-series categorical attention over three tokens is still a
# weak mechanism -- report it as what it is, not as cross-series attention.
for variant in cross_attention film_attention; do
    run_github transformer "$variant" || FAILED=1
done

# ---- Tier 2b: multidimensional (S4ND-style 2d/3d/4d) combo variants ----------
# gru/lstm + transformer/gpt (the latter only get the axial combo variants),
# plus mamba_nd (GPU/triton only, like `mamba` in Tier 1 — it was previously
# launched by hand, which left its published arms unreproducible from here).
#
# cross_attention_{2,3,4}d is a RANK label, not a mechanism: gru/lstm/
# transformer/gpt run axial cross-attention, mamba_nd runs per-axis Mamba-2
# scans, s4nd (Tier 4) runs separable per-axis S4. All three share the S4ND
# matched-information protocol — 2d embeds Commodity+Flow, 3d embeds Flow, 4d
# embeds nothing — so the comparison isolates the mechanism, not the inputs.
# matched_axial_{2,3,4}d is an alias for these, so do NOT run it as a separate
# tier — it trains the same models twice.
# mamba and s4 join the ACA family so every backbone family has an axial arm:
# recurrent (gru/lstm), attention (transformer/gpt) and SSM (mamba/s4). That
# also creates the matched pairs the multidimensional question needs --
# s4nd vs s4+ACA, and mamba_nd vs mamba+ACA -- where the sequence model is
# held identical and only the way the grid is injected changes.
for model in $EXP2_AXIAL_MODELS; do
    for variant in $EXP2_AXIAL_VARIANTS; do
        run_github "$model" "$variant" || FAILED=1
    done
done

# ---- Tier 3: classical baselines ---------------------------------------------
for b in $BASELINES; do
    run_baseline "$b" || FAILED=1
done

# ---- Tier 4: S4 family (real upstream repo) ----------------------------------
# Parity with the github side. No experiment config wires early stopping (the
# block in configs/callbacks/checkpoint.yaml is commented out), so it is added
# here -- registry.py maps "early_stopping" to Lightning's EarlyStopping and
# train.py instantiates everything under config.callbacks.
#
# patience is counted in VALIDATION CALLS, not epochs, which is why every s4nd
# experiment now sets val_check_interval: 1.0 -- at the old 0.5 a patience of N
# meant N/2 epochs here and N epochs on the github side, and s4nd also got two
# checkpoint-selection chances per epoch instead of one.
# Resolve the shared recipe once; every line is one Hydra override.
S4_TEXT="$("$PY" scripts/prepare_exp2.py --s4-overrides)" || exit 2
S4_COMMON=()
while IFS= read -r override; do S4_COMMON+=("$override"); done <<< "$S4_TEXT"
for exp in $EXP2_S4_EXPERIMENTS; do
    run_s4 "$exp" "${S4_COMMON[@]}" || FAILED=1
done

if [ "$FAILED" -ne 0 ]; then
    echo "[FAIL] Full test finished with one or more failed benchmark arms."
    exit 1
fi
echo "Full test complete. Artifacts in $RUNS/"
