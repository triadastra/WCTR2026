#!/usr/bin/env bash
# H20 box driver for the linear-head Transformer/GPT rerun and its SHAP.
# Nothing here runs unless invoked on the box; see docs/LINHEAD_RERUN.md.
#
#   bash scripts/linhead_box.sh check        # preflight only, starts nothing
#   bash scripts/linhead_box.sh train        # 136 runs in the background (resumable)
#   bash scripts/linhead_box.sh shap1        # Exp 1 SHAP, every linear-head run
#   bash scripts/linhead_box.sh shap2-flat   # Exp 2 SHAP, 1-D arms, seed 26, all lanes
#   bash scripts/linhead_box.sh shap2-nd     # Exp 2 SHAP, axial arms, seed 26, all lanes
#   bash scripts/linhead_box.sh shap2-sum    # rebuild the Exp 2 SHAP workbook
# shap2-flat and shap2-nd share one output folder: run them one after the other.
#   bash scripts/linhead_box.sh status
#
# The release lives in $WCTR_LINHEAD (code + data) and writes only under
# $WCTR_LINHEAD_OUT on persistent storage. The published production outputs
# are never written. SHAP settings mirror the published SHAP runs: Exp 1 as in
# wctr-shap-exp1 (every run), Exp 2 as in analysis/exp2_shap_seed26 (seed 26,
# every kept lane).
set -uo pipefail
WCTR_LINHEAD="${WCTR_LINHEAD:-/root/autodl-tmp/wctr-linhead}"
WCTR_LINHEAD_OUT="${WCTR_LINHEAD_OUT:-/root/autodl-fs/wctr-linhead}"
PY="${PY:-/root/autodl-tmp/wctr-runtime-20260922/bin/python}"
REPO="$WCTR_LINHEAD/repo"
cd "$REPO" || { echo "missing $REPO" >&2; exit 2; }

export DATA_DIR="$REPO" PY
export TMPDIR="$WCTR_LINHEAD/tmp" XDG_CACHE_HOME="$WCTR_LINHEAD/cache"
export TRITON_CACHE_DIR="$WCTR_LINHEAD/cache/triton" MPLCONFIGDIR="$WCTR_LINHEAD/cache/matplotlib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=16
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONHASHSEED=26 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTHONPATH PYTHONHOME SEEDS EXP1_SEEDS
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$WCTR_LINHEAD_OUT"

RUNS="$WCTR_LINHEAD_OUT/runs"          # becomes LINHEAD_ROOT (agg/, full/)
alive() { # name : its recorded PID is still that job (PIDs are reused after a restart)
  local f="$WCTR_LINHEAD_OUT/$1.pid" pid
  [ -f "$f" ] || return 1
  pid=$(cat "$f")
  [ -r "/proc/$pid/cmdline" ] && grep -qa -e "run_linhead.sh" -e "explain_exp" "/proc/$pid/cmdline"
}
background() { # name command...
  local name="$1"; shift
  local log="$WCTR_LINHEAD_OUT/$name.$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup "$@" > "$log" 2>&1 < /dev/null &
  echo "$!" > "$WCTR_LINHEAD_OUT/$name.pid"
  echo "$name started: PID $! log $log"
}
stamp_script() { # out-dir script...
  local out="$1"; shift
  mkdir -p "$out"; sha256sum "$@" > "$out/scripts.sha256"; cp "$@" "$out/"
  git rev-parse HEAD > "$out/git_commit" 2>/dev/null || true
}

case "${1:-check}" in
  check)
    "$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('cuda', torch.cuda.get_device_name(0))" || exit 2
    "$PY" scripts/check_legacy_data.py --data-dir "$DATA_DIR" >/dev/null || { echo "data audit failed" >&2; exit 2; }
    "$PY" -m pytest -q -p no:cacheprovider tests/test_linhead.py || exit 2
    bash scripts/run_linhead.sh --print-plan
    echo "check passed; nothing started" ;;
  train)
    # run_linhead.sh itself refuses a live lock and clears a stale one.
    alive train && { echo "training already running (PID $(cat "$WCTR_LINHEAD_OUT/train.pid"))" >&2; exit 2; }
    background train env LINHEAD_ROOT="$RUNS" RESUME="${RESUME:-1}" bash scripts/run_linhead.sh ;;
  shap1)
    O="$WCTR_LINHEAD_OUT/shap_exp1"; stamp_script "$O" scripts/explain_exp1.py
    background shap1 "$PY" -u scripts/explain_exp1.py --runs-dir "$RUNS/agg" --out "$O" --data-dir "$DATA_DIR" ;;
  shap2-flat)
    O="$WCTR_LINHEAD_OUT/shap_exp2_seed26"; stamp_script "$O" scripts/explain_exp2.py
    background shap2_flat "$PY" -u scripts/explain_exp2.py run --runs-dir "$RUNS/full" --out "$O" \
        --data-dir "$DATA_DIR" --kinds flat --seeds 26 --all-lanes ;;
  shap2-nd)
    O="$WCTR_LINHEAD_OUT/shap_exp2_seed26"; stamp_script "$O" scripts/explain_exp2.py
    background shap2_nd "$PY" -u scripts/explain_exp2.py run --runs-dir "$RUNS/full" --out "$O" \
        --data-dir "$DATA_DIR" --kinds combo --seeds 26 --combo-seeds 26 --all-lanes ;;
  shap2-sum)
    "$PY" scripts/explain_exp2.py summarise --out "$WCTR_LINHEAD_OUT/shap_exp2_seed26" ;;
  status)
    for f in "$WCTR_LINHEAD_OUT"/*.pid; do
      [ -e "$f" ] || continue
      alive "$(basename "$f" .pid)" && s=running || s=exited
      echo "$(basename "$f" .pid): $s"
    done
    echo "Exp 1 done: $(ls "$RUNS"/agg/*/metrics.json 2>/dev/null | wc -l)/100"
    echo "Exp 2 done: $(ls "$RUNS"/full/*/metrics.json 2>/dev/null | wc -l)/36" ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
