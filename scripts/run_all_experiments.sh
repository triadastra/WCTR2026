#!/usr/bin/env bash
# Canonical controller for all WCTR benchmark experiments.
#
# Paper run (the only mode that may rebuild results.xlsx):
#   bash scripts/run_all_experiments.sh
#
# Inspect the complete protocol without starting work:
#   bash scripts/run_all_experiments.sh --print-plan
#
# Small diagnostic run (never rebuilds the paper workbook):
#   PIPELINE_MODE=smoke SEEDS="42" EXP1_SEEDS="42" \
#     SWEEP_SEEDS="42" RUN_SWEEP=0 GPUS="0" \
#     bash scripts/run_all_experiments.sh
#
# The YAML files own model hyperparameters. This file owns experiment membership,
# seeds, scheduling, artifact validation, and the build gate. In paper mode the
# protocol variables below are immutable: overrides are rejected rather than
# silently producing a workbook from a different experiment.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

readonly EXPERIMENT_MANIFEST="config/experiment_manifest.env"
if [ ! -r "$EXPERIMENT_MANIFEST" ]; then
  echo "[preflight FAIL] missing $EXPERIMENT_MANIFEST" >&2
  exit 2
fi
# Validate the manifest as inert data before Bash is allowed to source it.
# This prevents shell expansion from being hidden in the one file deliberately
# shared by the Python collector and this controller.
PY="${PY:-${PYTHON:-python}}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[preflight FAIL] Python executable not found: $PY" >&2
  exit 2
fi
if ! "$PY" -m src.utils.experiment_manifest --repo . >/dev/null; then
  echo "[preflight FAIL] invalid or conflicting $EXPERIMENT_MANIFEST" >&2
  exit 2
fi
# shellcheck source=../config/experiment_manifest.env
source "$EXPERIMENT_MANIFEST"
readonly PAPER_SEEDS PAPER_EXP1_SEEDS PAPER_EXP1_BASELINE_SEED
readonly PAPER_EXP1_INPUT_LEN PAPER_EXP1_LAG_COUNT
readonly PAPER_EXP1_TRAIN_TARGET_START PAPER_EXP1_BATCH_SIZE
readonly PAPER_STATES PAPER_SOURCE_FINGERPRINT PAPER_SOURCE_START PAPER_SOURCE_END
readonly PAPER_TARGET_MEASURE PAPER_DUPLICATE_POLICY PAPER_COMMODITY_UNIVERSE PAPER_HS2_CODES
readonly PAPER_DATA_START PAPER_DATA_END
readonly PAPER_TRAIN_END PAPER_VAL_START PAPER_VAL_END PAPER_TEST_START
readonly PAPER_TEST_END PAPER_NORMALIZATION_METHOD
readonly PAPER_SWEEP_MODELS PAPER_WINDOWS PAPER_LAGS PAPER_SWEEP_BATCH
readonly PAPER_SWEEP_TRAIN_START BASIC_VARIANTS EXP1_NEURAL_MODELS
readonly EXP1_TREE_MODELS EXP2_BASIC_MODELS EXP2_COMBO_MODELS
readonly EXP2_AXIAL_MODELS EXP2_AXIAL_VARIANTS EXP2_S4_EXPERIMENTS BASELINES
readonly PAPER_MOVING_AVERAGE_WINDOW

PIPELINE_MODE="${PIPELINE_MODE:-paper}" # paper | smoke
SEEDS="${SEEDS:-$PAPER_SEEDS}"          # Exp 2 seeds
EXP1_SEEDS="${EXP1_SEEDS:-$PAPER_EXP1_SEEDS}"
SWEEP_SEEDS="${SWEEP_SEEDS:-$PAPER_SEEDS}"
SWEEP_MODELS="${SWEEP_MODELS:-$PAPER_SWEEP_MODELS}"
WINDOWS="${WINDOWS:-$PAPER_WINDOWS}"
LAGS="${LAGS:-$PAPER_LAGS}"
SWEEP_BATCH="${SWEEP_BATCH:-$PAPER_SWEEP_BATCH}"
SWEEP_TRAIN_START="${SWEEP_TRAIN_START:-$PAPER_SWEEP_TRAIN_START}"
DATA_DIR="${DATA_DIR:-.}"
GPUS="${GPUS:-0 1 2}"
RUN_SWEEP="${RUN_SWEEP:-1}"
RESUME="${RESUME:-0}"
CHECK_GPU="${CHECK_GPU:-1}"
export RESUME PY DATA_DIR
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

word_count() {
  # Intentional word splitting: protocol lists contain whitespace-delimited IDs.
  set -- $1
  echo "$#"
}

print_plan() {
  local n_exp1_seeds n_protocol_seeds n_exp2_arms n_exp3_cells
  n_exp1_seeds=$(word_count "$EXP1_SEEDS")
  n_protocol_seeds=$(word_count "$SEEDS")
  n_exp2_arms=$((
      $(word_count "$EXP2_BASIC_MODELS") * $(word_count "$BASIC_VARIANTS")
    + $(word_count "$EXP2_COMBO_MODELS") * 2
    + $(word_count "$EXP2_AXIAL_MODELS") * $(word_count "$EXP2_AXIAL_VARIANTS")
    + $(word_count "$BASELINES")
    + $(word_count "$EXP2_S4_EXPERIMENTS")
  ))
  n_exp3_cells=$((
      $(word_count "$SWEEP_MODELS") * $(word_count "$WINDOWS") * $(word_count "$LAGS")
  ))
  echo "Pipeline mode: $PIPELINE_MODE"
  echo "Exp 1: 14 neural/SSM arms + 4 stochastic tree arms x $n_exp1_seeds seeds; 3 deterministic baselines x 1 canonical run (seed $PAPER_EXP1_BASELINE_SEED)"
  echo "Exp 2: $n_exp2_arms arms x $n_protocol_seeds seeds"
  echo "Exp 3: $n_exp3_cells grid cells x $(word_count "$SWEEP_SEEDS") seeds"
  echo "Exp 2 seeds: $SEEDS"
  echo "Exp 1 seeds: $EXP1_SEEDS"
  echo "Exp 3: models=[$SWEEP_MODELS] windows=[$WINDOWS] lags=[$LAGS] seeds=[$SWEEP_SEEDS] batch=$SWEEP_BATCH target_start=$SWEEP_TRAIN_START"
}

case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --print-plan) print_plan; exit 0 ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

die() { echo "[preflight FAIL] $*" >&2; exit 2; }

require_choice() {
  local name="$1" value="$2"; shift 2
  local allowed
  for allowed in "$@"; do [ "$value" = "$allowed" ] && return 0; done
  die "$name must be one of: $* (got '$value')"
}

validate_integer_list() {
  local name="$1" values="$2" value seen=" "
  [ -n "$values" ] || die "$name must not be empty"
  for value in $values; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name contains a non-negative integer violation: '$value'"
    case "$seen" in *" $value "*) die "$name contains duplicate value '$value'" ;; esac
    seen="$seen$value "
  done
}

validate_paper_protocol() {
  [ "$SEEDS" = "$PAPER_SEEDS" ] || die "paper mode requires SEEDS='$PAPER_SEEDS'"
  [ "$EXP1_SEEDS" = "$PAPER_EXP1_SEEDS" ] || die "paper mode requires the canonical 25 EXP1_SEEDS"
  [ "$SWEEP_SEEDS" = "$PAPER_SEEDS" ] || die "paper mode requires SWEEP_SEEDS='$PAPER_SEEDS'"
  [ "$SWEEP_MODELS" = "$PAPER_SWEEP_MODELS" ] || die "paper mode requires SWEEP_MODELS='$PAPER_SWEEP_MODELS'"
  [ "$WINDOWS" = "$PAPER_WINDOWS" ] || die "paper mode requires WINDOWS='$PAPER_WINDOWS'"
  [ "$LAGS" = "$PAPER_LAGS" ] || die "paper mode requires LAGS='$PAPER_LAGS'"
  [ "$SWEEP_BATCH" = "$PAPER_SWEEP_BATCH" ] || die "paper mode requires SWEEP_BATCH=$PAPER_SWEEP_BATCH"
  [ "$SWEEP_TRAIN_START" = "$PAPER_SWEEP_TRAIN_START" ] || die "paper mode requires SWEEP_TRAIN_START=$PAPER_SWEEP_TRAIN_START"
}

preflight() {
  require_choice PIPELINE_MODE "$PIPELINE_MODE" paper smoke
  require_choice RUN_SWEEP "$RUN_SWEEP" 0 1
  require_choice RESUME "$RESUME" 0 1
  require_choice CHECK_GPU "$CHECK_GPU" 0 1
  validate_integer_list SEEDS "$SEEDS"
  validate_integer_list EXP1_SEEDS "$EXP1_SEEDS"
  validate_integer_list SWEEP_SEEDS "$SWEEP_SEEDS"
  validate_integer_list GPUS "$GPUS"
  validate_integer_list WINDOWS "$WINDOWS"
  validate_integer_list LAGS "$LAGS"
  [[ "$SWEEP_BATCH" =~ ^[1-9][0-9]*$ ]] || die "SWEEP_BATCH must be a positive integer"
  [ "$PIPELINE_MODE" = smoke ] || validate_paper_protocol

  command -v "$PY" >/dev/null 2>&1 || die "Python executable not found: $PY"
  command -v bash >/dev/null 2>&1 || die "bash is required"
  [ -f scripts/run_full_test.sh ] || die "run from a complete repository checkout"
  [ -f config/base.yaml ] || die "missing config/base.yaml"
  [ -d "$DATA_DIR" ] || die "missing data root: $DATA_DIR"
  # The canonical source checker validates the configured monthly release;
  # raw report folders are only needed by the legacy reproduction profile.
  "$PY" scripts/check_legacy_data.py --data-dir "$DATA_DIR" >/dev/null || \
      die "configured source fingerprint/domain/duplicate audit failed"
  "$PY" -m src.utils.experiment_manifest --repo . >/dev/null || \
      die "experiment manifest does not match the resolved Exp 1 YAML"
  # No download or replacement-panel construction: preserve the pinned CSVs.
  # Check the entire launch stack now, not after hundreds of Exp1 jobs.
  "$PY" scripts/prepare_exp2.py --check-runtime --data-dir "$DATA_DIR" || \
      die "full experiment runtime/recipe preflight failed"

  if [ "$CHECK_GPU" = 1 ]; then
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found (set CHECK_GPU=0 only for a deliberate non-GPU diagnostic)"
    local available=" " gpu
    while IFS= read -r gpu; do available="$available${gpu//[[:space:]]/} "; done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    for gpu in $GPUS; do
      case "$available" in *" $gpu "*) ;; *) die "GPU index $gpu is not available" ;; esac
    done
  fi
}

preflight
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
export DATA_DIR
# Resolve once so changing directories for the upstream S4 evaluator cannot
# reinterpret a repository-relative Python path.
PY="$(command -v "$PY")"
export PY
mkdir -p outputs/pipeline_runs

LOCK_DIR="outputs/.run_all_experiments.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  owner=$(sed -n '1p' "$LOCK_DIR/owner" 2>/dev/null || true)
  die "another pipeline holds $LOCK_DIR${owner:+ ($owner)}"
fi
echo "pid=$$ host=$(hostname) started=$(date -u +%FT%TZ)" > "$LOCK_DIR/owner"

ACTIVE_PIDS=()
# Give each background worker a process group. Killing only its shell PID
# leaves Python/CUDA processes and DataLoader descendants running after unlock.
set -m
INTERRUPTED=0
cleanup() {
  local rc=$?
  trap - EXIT
  # A second Ctrl-C/TERM must not abort cleanup halfway through its grace
  # period and strand the very descendants we are trying to stop.
  trap '' INT TERM HUP
  if [ "$INTERRUPTED" -eq 1 ]; then
    local pid active deadline
    if [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; then
      for pid in "${ACTIVE_PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
      deadline=$((SECONDS + 5))
      while :; do
        active=0
        for pid in "${ACTIVE_PIDS[@]}"; do
          if kill -0 -- "-$pid" 2>/dev/null; then active=1; fi
        done
        [ "$active" -eq 1 ] || break
        if [ "$SECONDS" -ge "$deadline" ]; then
          for pid in "${ACTIVE_PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
          break
        fi
        sleep 0.1
      done
      # A bare wait also waits for the tee process substitution on Linux.
      # tee needs this shell's output descriptors to close before it can exit.
      for pid in "${ACTIVE_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    fi
  fi
  rm -f "$LOCK_DIR/owner"
  rmdir "$LOCK_DIR" 2>/dev/null || true
  if [ -n "${RUN_META_DIR:-}" ] && [ -d "$RUN_META_DIR" ]; then
    {
      echo "ended_utc=$(date -u +%FT%TZ)"
      echo "exit_code=$rc"
      echo "interrupted=$INTERRUPTED"
    } >> "$RUN_META_DIR/manifest.txt"
  fi
  return "$rc"
}
trap cleanup EXIT
trap 'INTERRUPTED=1; exit 130' INT TERM HUP

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
RUN_META_DIR="outputs/pipeline_runs/$RUN_STAMP"
mkdir -p "$RUN_META_DIR"
exec > >(tee -a "$RUN_META_DIR/pipeline.log") 2>&1
if [ "$PIPELINE_MODE" = paper ]; then
  EXP1_RUNS_DIR="outputs/agg"
else
  # Diagnostics must never contaminate the directory the paper builder scans.
  EXP1_RUNS_DIR="$RUN_META_DIR/smoke_outputs/agg"
fi
mkdir -p "$EXP1_RUNS_DIR"
{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "pid=$$"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo true || echo false)"
  echo "python=$PY"
  echo "data_dir=$DATA_DIR"
  echo "gpus=$GPUS"
  echo "resume=$RESUME"
  echo "experiment_manifest=$EXPERIMENT_MANIFEST"
  echo "exp1_runs_dir=$EXP1_RUNS_DIR"
  print_plan
} > "$RUN_META_DIR/manifest.txt"
{
  uname -a 2>/dev/null || true
  "$PY" --version 2>&1 || true
  "$PY" -m pip --version 2>&1 || true
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi 2>&1; fi
} > "$RUN_META_DIR/environment.txt"
"$PY" -m pip freeze > "$RUN_META_DIR/requirements.freeze.txt" 2>&1 || true

GPU_ARR=($GPUS)
NG=${#GPU_ARR[@]}
FAILED=0
echo "=== WCTR experiment pipeline | run=$RUN_STAMP | $(date -u +%FT%TZ) ==="
print_plan
echo "Monitor: $PY scripts/train_status.py --watch"

wait_for_active_jobs() {
  local pid rc=0
  [ "${#ACTIVE_PIDS[@]}" -gt 0 ] || return 0
  while [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; do
    pid=${ACTIVE_PIDS[0]}
    wait "$pid" || rc=1
    ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
  done
  return "$rc"
}

launch_seed_jobs() { # launch_seed_jobs <worker-function> <seed-list> <log-prefix>
  local worker="$1" seeds="$2" prefix="$3" seed gpu i=0
  for seed in $seeds; do
    gpu=${GPU_ARR[$((i % NG))]}
    echo "[launch] $prefix seed $seed -> GPU $gpu"
    "$worker" "$seed" "$gpu" > "$RUN_META_DIR/${prefix}_seed${seed}.log" 2>&1 &
    ACTIVE_PIDS+=("$!")
    i=$((i + 1))
    if (( i % NG == 0 )); then wait_for_active_jobs || FAILED=1; fi
  done
  wait_for_active_jobs || FAILED=1
}

run_exp2_seed() {
  local seed="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_full_test.sh "$seed"
}

run_exp1_model() { # model variant seed out
  local model="$1" variant="$2" seed="$3" out="$4" ckpt
  if "$PY" scripts/run_is_current.py "$out/metrics.json" --aggregate \
      --exp1-manifest "$EXPERIMENT_MANIFEST" \
      --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
      --train-target-start "$PAPER_EXP1_TRAIN_TARGET_START" --expected-seed "$seed" \
      --expected-model "$model" --expected-variant "$variant"; then
    echo "[Exp1 skip] $out"
    return 0
  fi
  mkdir -p "$out"
  # metrics.json is the completion marker.  Never leave a stale marker visible
  # while a replacement checkpoint is being trained.
  rm -f "$out/metrics.json"
  local resume_flag=""
  [ "$RESUME" = 1 ] && resume_flag="--resume"
  local train_rc=0
  if [ "$RESUME" = 1 ]; then
    "$PY" scripts/train.py --model "$model" --variant "$variant" --seed "$seed" \
        --data-dir "$DATA_DIR" --out-dir "$out" --aggregate \
        --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
        --batch-size "$PAPER_EXP1_BATCH_SIZE" "$resume_flag" \
        >> "$out/train.log" 2>&1 || train_rc=$?
  else
    "$PY" scripts/train.py --model "$model" --variant "$variant" --seed "$seed" \
        --data-dir "$DATA_DIR" --out-dir "$out" --aggregate \
        --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
        --batch-size "$PAPER_EXP1_BATCH_SIZE" \
        > "$out/train.log" 2>&1 || train_rc=$?
  fi
  if [ "$train_rc" -ne 0 ]; then
    echo "[Exp1 FAIL] training $model/$variant seed $seed; inference suppressed"
    return 1
  fi
  ckpt="$out/best.pth"
  case "$model" in xgboost|lightgbm|random_forest) ckpt="$out/best_xgb" ;; esac
  if ! "$PY" scripts/inference.py --model "$model" --variant "$variant" --seed "$seed" \
      --data-dir "$DATA_DIR" --out-dir "$out" --aggregate --normalized --device cuda \
      --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
      --batch-size "$PAPER_EXP1_BATCH_SIZE" \
      --checkpoint "$ckpt" > "$out/inference.log" 2>&1; then
    echo "[Exp1 FAIL] inference $model/$variant seed $seed"
    return 1
  fi
  if ! "$PY" scripts/run_is_current.py "$out/metrics.json" --aggregate \
      --exp1-manifest "$EXPERIMENT_MANIFEST" \
      --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
      --train-target-start "$PAPER_EXP1_TRAIN_TARGET_START" --expected-seed "$seed" \
      --expected-model "$model" --expected-variant "$variant"; then
    echo "[Exp1 FAIL] post-inference artifact validation $model/$variant seed $seed"
    return 1
  fi
}

run_exp1_seed() {
  local seed="$1" gpu="$2" model variant baseline out failed=0
  export CUDA_VISIBLE_DEVICES="$gpu"
  for model in $EXP1_NEURAL_MODELS; do
    for variant in $BASIC_VARIANTS; do
      out="$EXP1_RUNS_DIR/${model}_${variant}_seed${seed}"
      run_exp1_model "$model" "$variant" "$seed" "$out" || failed=1
    done
  done
  for model in $EXP1_TREE_MODELS; do
    for variant in $BASIC_VARIANTS; do
      out="$EXP1_RUNS_DIR/${model}_${variant}_seed${seed}"
      run_exp1_model "$model" "$variant" "$seed" "$out" || failed=1
    done
  done
  case " $PAPER_EXP1_BASELINE_SEED " in
    *" $seed "*)
      for baseline in $BASELINES; do
        local baseline_rc=0
        out="$EXP1_RUNS_DIR/${baseline}_none_seed${seed}"
        if "$PY" scripts/run_is_current.py "$out/metrics.json" --aggregate \
            --exp1-manifest "$EXPERIMENT_MANIFEST" \
            --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
            --train-target-start "$PAPER_EXP1_TRAIN_TARGET_START" --expected-seed "$seed" \
            --expected-model "$baseline" --expected-variant none; then
          echo "[Exp1 skip] $out"
        else
          mkdir -p "$out"
          rm -f "$out/metrics.json"
          "$PY" scripts/run_baseline.py --model "$baseline" --seed "$seed" \
              --data-dir "$DATA_DIR" --out-dir "$out" --aggregate --normalized \
              --window "$PAPER_MOVING_AVERAGE_WINDOW" \
              --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
              --batch-size "$PAPER_EXP1_BATCH_SIZE" \
              > "$out/run.log" 2>&1 || baseline_rc=$?
          if [ "$baseline_rc" -eq 0 ]; then
            "$PY" scripts/run_is_current.py "$out/metrics.json" --aggregate \
                --exp1-manifest "$EXPERIMENT_MANIFEST" \
                --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
                --train-target-start "$PAPER_EXP1_TRAIN_TARGET_START" --expected-seed "$seed" \
                --expected-model "$baseline" --expected-variant none || baseline_rc=1
          fi
          [ "$baseline_rc" -eq 0 ] || failed=1
        fi
      done
      ;;
  esac
  return "$failed"
}

run_exp3_cell() {
  local model="$1" window="$2" lag="$3" seed="$4" gpu="$5" variant=embeddings
  local out="outputs/sweep/${model}_${variant}_w${window}_l${lag}_seed${seed}"
  local protocol_flag=""
  [ "$PIPELINE_MODE" = paper ] && protocol_flag="--exp3"
  export CUDA_VISIBLE_DEVICES="$gpu"
  if "$PY" scripts/run_is_current.py "$out/metrics.json" ${protocol_flag:+"$protocol_flag"} \
      --input-len "$window" --lag-count "$lag" \
      --train-target-start "$SWEEP_TRAIN_START" --expected-seed "$seed" \
      --expected-model "$model" --expected-variant "$variant"; then
    echo "[Exp3 skip] $out"
    return 0
  fi
  mkdir -p "$out"
  # The completion marker is published only by successful inference. Keeping a
  # stale marker visible while retraining lets monitors and later invocations
  # mistake an in-progress replacement for a current result.
  rm -f "$out/metrics.json"
  local resume_flag=""
  [ "$RESUME" = 1 ] && resume_flag="--resume"
  if ! "$PY" scripts/train.py --exp3 --model "$model" --variant "$variant" --seed "$seed" \
      --data-dir "$DATA_DIR" --out-dir "$out" --input-len "$window" --lag-count "$lag" \
      --train-target-start "$SWEEP_TRAIN_START" --batch-size "$SWEEP_BATCH" \
      ${resume_flag:+"$resume_flag"} > "$out/train.log" 2>&1; then
    echo "[Exp3 FAIL] training $model w=$window lag=$lag seed=$seed; inference suppressed"
    return 1
  fi
  if ! "$PY" scripts/inference.py --exp3 --model "$model" --variant "$variant" --seed "$seed" \
      --data-dir "$DATA_DIR" --out-dir "$out" --input-len "$window" --lag-count "$lag" \
      --train-target-start "$SWEEP_TRAIN_START" --normalized --device cuda \
      --checkpoint "$out/best.pth" > "$out/inference.log" 2>&1; then
    echo "[Exp3 FAIL] inference $model w=$window lag=$lag seed=$seed"
    return 1
  fi
  "$PY" scripts/run_is_current.py "$out/metrics.json" ${protocol_flag:+"$protocol_flag"} \
      --input-len "$window" --lag-count "$lag" \
      --train-target-start "$SWEEP_TRAIN_START" --expected-seed "$seed" \
      --expected-model "$model" --expected-variant "$variant"
}

run_exp3_grid() {
  local seed model window lag gpu i=0
  for seed in $SWEEP_SEEDS; do
    for model in $SWEEP_MODELS; do
      for window in $WINDOWS; do
        for lag in $LAGS; do
          gpu=${GPU_ARR[$((i % NG))]}
          run_exp3_cell "$model" "$window" "$lag" "$seed" "$gpu" \
              > "$RUN_META_DIR/exp3_${model}_w${window}_l${lag}_seed${seed}.log" 2>&1 &
          ACTIVE_PIDS+=("$!")
          i=$((i + 1))
          if (( i % NG == 0 )); then wait_for_active_jobs || FAILED=1; fi
        done
      done
    done
  done
  wait_for_active_jobs || FAILED=1
}

verify_artifact() { # metrics-file and run_is_current arguments
  local metrics="$1"; shift
  if ! "$PY" scripts/run_is_current.py "$metrics" "$@"; then
    echo "[verify FAIL] $metrics"
    return 1
  fi
}

verify_no_unexpected_metrics() { # root, whitespace-delimited expected dirs
  local root="$1" expected="$2" metrics dir rc=0
  for metrics in "$root"/*/metrics.json; do
    [ -e "$metrics" ] || continue
    dir=${metrics%/metrics.json}
    case " $expected " in
      *" $dir "*) ;;
      *) echo "[verify FAIL] unexpected result directory: $dir"; rc=1 ;;
    esac
  done
  return "$rc"
}

s4_expected_identity() { # experiment -> "model variant"
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

verify_paper_outputs() {
  local seed model variant baseline exp window lag path rc=0 identity
  local expected_agg="" expected_full="" expected_sweep=""

  for seed in $EXP1_SEEDS; do
    for model in $EXP1_NEURAL_MODELS $EXP1_TREE_MODELS; do for variant in $BASIC_VARIANTS; do
      path="$EXP1_RUNS_DIR/${model}_${variant}_seed${seed}"; expected_agg="$expected_agg $path"
      verify_artifact "$path/metrics.json" --aggregate \
          --exp1-manifest "$EXPERIMENT_MANIFEST" \
          --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
          --train-target-start "$PAPER_EXP1_TRAIN_TARGET_START" \
          --expected-seed "$seed" --expected-model "$model" --expected-variant "$variant" || rc=1
    done; done
  done
  for seed in $PAPER_EXP1_BASELINE_SEED; do
    for baseline in $BASELINES; do
      path="$EXP1_RUNS_DIR/${baseline}_none_seed${seed}"; expected_agg="$expected_agg $path"
      verify_artifact "$path/metrics.json" --aggregate \
          --exp1-manifest "$EXPERIMENT_MANIFEST" \
          --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
          --train-target-start "$PAPER_EXP1_TRAIN_TARGET_START" \
          --expected-seed "$seed" --expected-model "$baseline" --expected-variant none || rc=1
    done
  done

  for seed in $SEEDS; do
    for model in $EXP2_BASIC_MODELS; do for variant in $BASIC_VARIANTS; do
      path="outputs/full/${model}_${variant}_seed${seed}"; expected_full="$expected_full $path"
      verify_artifact "$path/metrics.json" --exp2 --expected-seed "$seed" \
          --expected-model "$model" --expected-variant "$variant" || rc=1
    done; done
    for model in $EXP2_COMBO_MODELS; do for variant in cross_attention film_attention; do
      path="outputs/full/${model}_${variant}_seed${seed}"; expected_full="$expected_full $path"
      verify_artifact "$path/metrics.json" --exp2 --expected-seed "$seed" \
          --expected-model "$model" --expected-variant "$variant" || rc=1
    done; done
    for model in $EXP2_AXIAL_MODELS; do for variant in $EXP2_AXIAL_VARIANTS; do
      path="outputs/full/${model}_${variant}_seed${seed}"; expected_full="$expected_full $path"
      verify_artifact "$path/metrics.json" --exp2 --expected-seed "$seed" \
          --expected-model "$model" --expected-variant "$variant" || rc=1
    done; done
    for baseline in $BASELINES; do
      path="outputs/full/${baseline}_none_seed${seed}"; expected_full="$expected_full $path"
      verify_artifact "$path/metrics.json" --exp2 --expected-seed "$seed" \
          --expected-model "$baseline" --expected-variant none || rc=1
    done
    for exp in $EXP2_S4_EXPERIMENTS; do
      path="outputs/full/s4_${exp}_seed${seed}"; expected_full="$expected_full $path"
      identity="$(s4_expected_identity "$exp")" || { rc=1; continue; }
      read -r model variant <<< "$identity"
      verify_artifact "$path/metrics.json" --exp2 --expected-seed "$seed" \
          --expected-model "$model" --expected-variant "$variant" || rc=1
    done
  done

  for seed in $SWEEP_SEEDS; do for model in $SWEEP_MODELS; do
    for window in $WINDOWS; do for lag in $LAGS; do
      path="outputs/sweep/${model}_embeddings_w${window}_l${lag}_seed${seed}"
      expected_sweep="$expected_sweep $path"
      verify_artifact "$path/metrics.json" --exp3 --input-len "$window" --lag-count "$lag" \
          --train-target-start "$SWEEP_TRAIN_START" --expected-seed "$seed" \
          --expected-model "$model" --expected-variant embeddings || rc=1
    done; done
  done; done

  verify_no_unexpected_metrics "$EXP1_RUNS_DIR" "$expected_agg" || rc=1
  verify_no_unexpected_metrics outputs/full "$expected_full" || rc=1
  # Experiment 3's Python builder explicitly excludes non-canonical smoke dirs.
  [ "$rc" -eq 0 ] || return 1
  echo "[verify] exact paper manifest is complete and current"
}

abort_if_failed() {
  local phase="$1"
  if [ "$FAILED" -ne 0 ]; then
    echo "[FAIL] $phase failed; later experiments and workbook build were not started"
    exit 1
  fi
}

# Phase 1: aggregate experiment. Neural/SSM and subsampled tree arms use 25
# seeds; closed-form deterministic baselines are emitted once at the canonical
# Exp 1 baseline seed.
launch_seed_jobs run_exp1_seed "$EXP1_SEEDS" exp1
abort_if_failed "Experiment 1"

# Phase 2: per-combination benchmark. One seed owns one GPU and runs its arms
# serially; the upstream S4/S4ND evaluator completes this experiment.
launch_seed_jobs run_exp2_seed "$SEEDS" exp2
abort_if_failed "Experiment 2"

# Each run_full_test worker already evaluated and verified its S4 arms.

# Phase 3: sensitivity grid. RUN_SWEEP=0 reuses existing artifacts but the paper
# build gate still demands the complete canonical grid.
if [ "$RUN_SWEEP" = 1 ]; then run_exp3_grid; fi
abort_if_failed "Experiment 3"

if [ "$PIPELINE_MODE" = smoke ]; then
  echo "=== SMOKE RUN COMPLETE: paper workbook intentionally not built ==="
  exit 0
fi

if ! verify_paper_outputs; then
  echo "[FAIL] exact experiment manifest verification failed; no workbook was built"
  exit 1
fi

"$PY" scripts/collect_results.py --runs-dir outputs/full || exit 1
"$PY" scripts/build_paper_results.py --repo . --out results.xlsx || exit 1
"$PY" scripts/build_graphs.py || { echo "[FAIL] graph generation failed; workbook is valid but reporting is incomplete"; exit 1; }
echo "completed_utc=$(date -u +%FT%TZ)" >> "$RUN_META_DIR/manifest.txt"
echo "=== DONE: verified Experiments 1-3 and rebuilt results.xlsx | $(date -u +%FT%TZ) ==="
