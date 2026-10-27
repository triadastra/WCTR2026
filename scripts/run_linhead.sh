#!/usr/bin/env bash
# Linear-head rerun of the Exp 1/2 Transformer and GPT arms (136 runs).
#
# The published transformer/gpt arms use MLP heads; every other neural arm
# reads out through Linear(128 -> 2). This retrains exactly those arms with the
# linear head under the names transformer_linhead / gpt_linhead and writes to
# outputs/linhead/{agg,full}. Published outputs/agg and outputs/full are never
# read, written or skipped over, so the paper gate is unaffected.
#
#   bash scripts/run_linhead.sh                  # Exp 1 then Exp 2
#   bash scripts/run_linhead.sh --print-plan     # roster only
#   PHASES="exp2" bash scripts/run_linhead.sh    # one phase
#   RESUME=1 bash scripts/run_linhead.sh         # continue killed runs
#
# Runs are skipped only when run_is_current.py --linhead accepts them, so a
# relaunch after an interruption retrains only what is missing or stale.
# Diagnostics: LINHEAD_ROOT=<dir> SEEDS="26" EXP1_SEEDS="104" DEVICE=cpu
# write elsewhere and use fewer seeds; they are never a paper result. Every
# run uses the canonical recipe, so the --linhead gate applies unchanged; for
# a shortened-epoch smoke call scripts/train.py --epochs directly instead.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${PY:-${PYTHON:-python}}"
command -v "$PY" >/dev/null 2>&1 || { echo "[preflight FAIL] Python not found: $PY" >&2; exit 2; }
"$PY" -m src.utils.experiment_manifest --repo . >/dev/null || exit 2
# shellcheck source=../config/experiment_manifest.env
source config/experiment_manifest.env

PHASES="${PHASES:-exp1 exp2}"
SEEDS="${SEEDS:-$PAPER_SEEDS}"
EXP1_SEEDS="${EXP1_SEEDS:-$PAPER_EXP1_SEEDS}"
LINHEAD_ROOT="${LINHEAD_ROOT:-outputs/linhead}"
DATA_DIR="${DATA_DIR:-.}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-0}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 DATA_DIR

# Same membership as the paper roster, renamed: every transformer/gpt arm.
EXP1_MODELS="transformer_linhead gpt_linhead"
EXP2_ARMS="$("$PY" -c '
from src.utils.linhead_protocol import exp2_runs
seen = []
for r in exp2_runs():
    if (r.model, r.variant) not in seen: seen.append((r.model, r.variant))
print(" ".join(f"{m}:{v}" for m, v in seen))')" || exit 2

if [ "${1:-}" = "--print-plan" ]; then
  "$PY" - <<'EOF'
from src.utils.linhead_protocol import exp1_runs, exp2_runs
e1, e2 = exp1_runs(), exp2_runs()
print(f"Exp 1: {len(e1)} runs  " + ", ".join(sorted({f'{r.model}_{r.variant}' for r in e1})))
print(f"Exp 2: {len(e2)} runs  " + ", ".join(dict.fromkeys(f'{r.model}_{r.variant}' for r in e2)))
print(f"Total: {len(e1) + len(e2)}")
EOF
  exit 0
fi

if [ "$LINHEAD_ROOT" = outputs/linhead ] && \
   { [ "$SEEDS" != "$PAPER_SEEDS" ] || [ "$EXP1_SEEDS" != "$PAPER_EXP1_SEEDS" ]; }; then
  echo "[preflight FAIL] reduced seeds must write to a LINHEAD_ROOT other than outputs/linhead" >&2
  exit 2
fi
"$PY" scripts/check_legacy_data.py --data-dir "$DATA_DIR" >/dev/null || {
  echo "[preflight FAIL] configured source fingerprint/domain/duplicate audit failed" >&2; exit 2; }
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

mkdir -p "$LINHEAD_ROOT"
LOCK_DIR="$LINHEAD_ROOT/.run_linhead.lock"
# A killed controller (container stop, kill -9) never runs its EXIT trap and
# leaves the lock behind. The lock records host and PID; it is stale, and is
# cleared, only when it was taken on this host and that PID is no longer a
# run_linhead.sh (a restarted container may reuse the number for anything).
lock_is_live() {
  local host pid
  read -r host pid < "$LOCK_DIR/owner" 2>/dev/null || return 0   # no owner yet: assume live
  [ "$host" = "$(hostname)" ] || return 0                        # another host: never touch
  grep -qa "run_linhead.sh" "/proc/$pid/cmdline" 2>/dev/null || ps -p "$pid" -o args= 2>/dev/null | grep -q "run_linhead.sh"
}
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [ -f "$LOCK_DIR/owner" ] && ! lock_is_live; then
    echo "[lock] clearing stale lock left by $(cat "$LOCK_DIR/owner")"
    rm -f "$LOCK_DIR/owner"; rmdir "$LOCK_DIR" 2>/dev/null
    mkdir "$LOCK_DIR" 2>/dev/null || { echo "[FAIL] could not take $LOCK_DIR" >&2; exit 2; }
  else
    echo "[FAIL] another run_linhead.sh holds $LOCK_DIR ($(cat "$LOCK_DIR/owner" 2>/dev/null || echo 'owner unknown; remove it by hand only if no run_linhead.sh is running'))" >&2
    exit 2
  fi
fi
echo "$(hostname) $$" > "$LOCK_DIR/owner"
trap 'rm -f "$LOCK_DIR/owner"; rmdir "$LOCK_DIR"' EXIT
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
LOG="$LINHEAD_ROOT/pipeline_runs/$STAMP.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== linear-head rerun | $STAMP | git $(git rev-parse --short HEAD 2>/dev/null) dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo true || echo false)"
echo "phases=[$PHASES] root=$LINHEAD_ROOT device=$DEVICE resume=$RESUME"

RESUME_ARGS=(); [ "$RESUME" = 1 ] && RESUME_ARGS=(--resume)
FAILED=0

exp1_current() { # metrics model variant seed
  "$PY" scripts/run_is_current.py "$1" --linhead --aggregate \
      --exp1-manifest config/experiment_manifest.env \
      --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
      --train-target-start "$PAPER_EXP1_TRAIN_TARGET_START" --expected-seed "$4" \
      --expected-model "$2" --expected-variant "$3"
}

run_exp1() { # model variant seed
  local model="$1" variant="$2" seed="$3"
  local out="$LINHEAD_ROOT/agg/${model}_${variant}_seed${seed}"
  if exp1_current "$out/metrics.json" "$model" "$variant" "$seed"; then
    echo "[Exp1 skip] $out"; return 0
  fi
  mkdir -p "$out"; rm -f "$out/metrics.json"
  [ "$RESUME" = 1 ] || : > "$out/train.log"   # resume keeps the earlier epochs
  echo "[Exp1 train] $model/$variant seed $seed"
  "$PY" scripts/train.py --model "$model" --variant "$variant" --seed "$seed" \
      --data-dir "$DATA_DIR" --out-dir "$out" --aggregate \
      --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
      --batch-size "$PAPER_EXP1_BATCH_SIZE" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
      >> "$out/train.log" 2>&1 || { echo "[Exp1 FAIL] training $model/$variant seed $seed"; return 1; }
  "$PY" scripts/inference.py --model "$model" --variant "$variant" --seed "$seed" \
      --data-dir "$DATA_DIR" --out-dir "$out" --aggregate --normalized --device "$DEVICE" \
      --input-len "$PAPER_EXP1_INPUT_LEN" --lag-count "$PAPER_EXP1_LAG_COUNT" \
      --batch-size "$PAPER_EXP1_BATCH_SIZE" --checkpoint "$out/best.pth" \
      > "$out/inference.log" 2>&1 || { echo "[Exp1 FAIL] inference $model/$variant seed $seed"; return 1; }
  exp1_current "$out/metrics.json" "$model" "$variant" "$seed" || {
    echo "[Exp1 FAIL] artifact validation $model/$variant seed $seed"; return 1; }
  echo "[Exp1 ok] $out"
}

exp2_current() { # metrics model variant seed
  "$PY" scripts/run_is_current.py "$1" --linhead --exp2 --input-len 36 --lag-count 12 \
      --train-target-start null --expected-seed "$4" --expected-model "$2" --expected-variant "$3"
}

run_exp2() { # model variant seed
  local model="$1" variant="$2" seed="$3"
  local out="$LINHEAD_ROOT/full/${model}_${variant}_seed${seed}"
  if exp2_current "$out/metrics.json" "$model" "$variant" "$seed"; then
    echo "[Exp2 skip] $out"; return 0
  fi
  mkdir -p "$out"; rm -f "$out/metrics.json"
  [ "$RESUME" = 1 ] || : > "$out/train.log"   # resume keeps the earlier epochs
  echo "[Exp2 train] $model/$variant seed $seed"
  "$PY" scripts/train.py --model "$model" --variant "$variant" --seed "$seed" \
      --data-dir "$DATA_DIR" --out-dir "$out" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
      >> "$out/train.log" 2>&1 || { echo "[Exp2 FAIL] training $model/$variant seed $seed"; return 1; }
  "$PY" scripts/inference.py --model "$model" --variant "$variant" --seed "$seed" \
      --checkpoint "$out/best.pth" --device "$DEVICE" \
      --data-dir "$DATA_DIR" --out-dir "$out" --normalized \
      > "$out/inference.log" 2>&1 || { echo "[Exp2 FAIL] inference $model/$variant seed $seed"; return 1; }
  exp2_current "$out/metrics.json" "$model" "$variant" "$seed" || {
    echo "[Exp2 FAIL] artifact validation $model/$variant seed $seed"; return 1; }
  echo "[Exp2 ok] $out"
}

for phase in $PHASES; do
  case "$phase" in
    exp1)
      for seed in $EXP1_SEEDS; do for model in $EXP1_MODELS; do for variant in $BASIC_VARIANTS; do
        run_exp1 "$model" "$variant" "$seed" || FAILED=1
      done; done; done ;;
    exp2)
      for seed in $SEEDS; do for arm in $EXP2_ARMS; do
        run_exp2 "${arm%%:*}" "${arm#*:}" "$seed" || FAILED=1
      done; done ;;
    *) echo "[FAIL] unknown phase $phase (expected exp1, exp2)"; exit 2 ;;
  esac
done

if [ "$FAILED" -ne 0 ]; then
  echo "[FAIL] one or more linear-head runs failed; rerun the same command to retry them"
  exit 1
fi
echo "=== DONE: linear-head runs current under $LINHEAD_ROOT | $(date -u +%FT%TZ) ==="
