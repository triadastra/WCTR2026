#!/usr/bin/env bash
# Box chain for the linear-head rerun (revised 2026-10-16 per the user):
#   wait for training -> Exp 1 SHAP (all 100) -> Exp 2 SHAP flat arms (seed 26,
#   all lanes) -> the two published Transformer flat arms that failed the
#   1e-4 gate on the Mac (seed 26, all lanes, on GPU) -> Exp 2 summary.
# The axial (shap2-nd) step is deliberately NOT run: on hold until the user
# picks lanes/models for both published and linear-head axial arms.
# Safe to rerun: SHAP skips runs whose results already exist. The chain stops
# at the first failing step and exits with its status; ALL DONE means every
# step succeeded.
set -u
O=/root/autodl-fs/wctr-linhead
REPO=/root/autodl-tmp/wctr-linhead/repo
PY=/root/autodl-tmp/wctr-runtime-20260922/bin/python
PUB=/root/autodl-fs/wctr-20260923-pr27/production-outputs/full
cd "$REPO" || exit 2
export DATA_DIR="$REPO" TMPDIR=/root/autodl-tmp/wctr-linhead/tmp
export XDG_CACHE_HOME=/root/autodl-tmp/wctr-linhead/cache MPLCONFIGDIR=/root/autodl-tmp/wctr-linhead/cache/matplotlib
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=26 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR"
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$O/after_train.log"; }
step() { # name command... : run, log, and stop the chain on failure
  local name="$1"; shift
  "$@"; local rc=$?
  if [ "$rc" -ne 0 ]; then log "$name FAILED (exit $rc); chain stopped"; exit "$rc"; fi
  log "$name ok"
}
GATE_FAILS=0
gates() { # out-dir : count runs whose final SHAP status is not ok (recorded, not fatal)
  local n   # a run that is retried with more nodes appends a new row; the last row counts
  n=$("$PY" -c 'import json, sys
last = {}
for line in open(sys.argv[1]):
    r = json.loads(line); last[r["run"]] = r["status"]
print(sum(v != "ok" for v in last.values()))' "$1/checks.jsonl" 2>/dev/null) || n=1
  log "$(basename "$1"): ${n} run(s) did not pass the SHAP gates (see $1/checks.jsonl)"
  GATE_FAILS=$((GATE_FAILS + n))
}
stamp() { mkdir -p "$1"; sha256sum scripts/explain_exp1.py scripts/explain_exp2.py > "$1/scripts.sha256"; git rev-parse HEAD > "$1/git_commit"; }

# 1. Training: wait for a running controller, otherwise (re)start it and wait.
# Match the PID to run_linhead.sh: after a container restart the old number
# can belong to an unrelated process, which a bare kill -0 would wait on forever.
train_alive() { [ -f "$O/train.pid" ] && grep -qa "run_linhead.sh" "/proc/$(cat "$O/train.pid")/cmdline" 2>/dev/null; }
if train_alive; then
  log "waiting for training PID $(cat "$O/train.pid")"
  while train_alive; do sleep 60; done
else
  log "no training running; resuming"
  LINHEAD_ROOT="$O/runs" RESUME=1 PY="$PY" bash scripts/run_linhead.sh > "$O/train.$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
  rc=$?; [ "$rc" -eq 0 ] || { log "training exited $rc; SHAP not started"; exit "$rc"; }
fi
TLOG=$(ls -t "$O"/train.*.log | head -1)
grep -q "=== DONE: linear-head runs current" "$TLOG" || { log "training not clean ($TLOG); SHAP not started"; exit 1; }
log "training done"

# 2. Exp 1 SHAP, every linear-head run.
stamp "$O/shap_exp1"
step shap1 sh -c '"$0" -u scripts/explain_exp1.py --runs-dir "$1/runs/agg" --out "$1/shap_exp1" --data-dir "$2" > "$1/shap_exp1/run.log" 2>&1' "$PY" "$O" "$DATA_DIR"

# explain_exp1 ends its log with "<ok>/<total> runs passed both gates".
read -r ok total < <(sed -n 's#^\([0-9]*\)/\([0-9]*\) runs passed both gates$#\1 \2#p' "$O/shap_exp1/run.log" | tail -1)
log "shap_exp1: ${ok:-?}/${total:-?} runs passed both gates"
[ -n "${ok:-}" ] && [ "$ok" = "$total" ] || GATE_FAILS=$((GATE_FAILS + 1))

# 3. Exp 2 SHAP, linear-head flat arms.
S2="$O/shap_exp2_seed26"; stamp "$S2"
step shap2-flat sh -c '"$0" -u scripts/explain_exp2.py run --runs-dir "$1/runs/full" --out "$2" --data-dir "$3" --kinds flat --seeds 26 --all-lanes >> "$2/run.log" 2>&1' "$PY" "$O" "$S2" "$DATA_DIR"
gates "$S2"
step shap2-flat-summary sh -c '"$0" scripts/explain_exp2.py summarise --out "$1" >> "$1/run.log" 2>&1' "$PY" "$S2"

# 4. Published Transformer flat arms that failed the gate on the Mac.
P="$O/shap_exp2_published_transformer_seed26"; stamp "$P"
step published-transformer-flat sh -c '"$0" -u scripts/explain_exp2.py run --runs-dir "$1" --out "$2" --data-dir "$3" --kinds flat --seeds 26 --all-lanes --only transformer_embeddings_seed26 transformer_cross_attention_seed26 >> "$2/run.log" 2>&1' "$PY" "$PUB" "$P" "$DATA_DIR"
gates "$P"
step published-transformer-summary sh -c '"$0" scripts/explain_exp2.py summarise --out "$1" >> "$1/run.log" 2>&1' "$PY" "$P"

if [ "$GATE_FAILS" -eq 0 ]; then
  log "ALL DONE: every step succeeded and every run passed its SHAP gates (axial shap2-nd on hold by user)"
else
  log "DONE WITH SHAP GATE FAILURES ($GATE_FAILS); see shap_exp1/run.log and checks.jsonl (axial shap2-nd on hold by user)"
  exit 3
fi
