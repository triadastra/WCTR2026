#!/usr/bin/env bash
# GPU-only part of the published Exp 2 1-D SHAP (seed 26, every kept lane).
#
# The Mac run (analysis/exp2_shap_seed26) covered every published 1-D arm it
# could: Mamba needs triton (CUDA), the hydra S4 arms were not run there, and
# transformer_{embeddings,cross_attention} failed the 1e-4 prediction gate on
# every Mac backend. This explains exactly those eight runs on the H20 with
# the same settings. The S4ND grid arms and every axial arm stay on hold.
#
#   nohup bash scripts/box_published_1d_shap.sh > <out>/driver.out 2>&1 &
#
# Read-only on the production outputs. Safe to rerun: finished runs are skipped.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/root/autodl-tmp/wctr-runtime-20260922/bin/python}"
PUB="${PUB:-/root/autodl-fs/wctr-20260923-pr27/production-outputs/full}"
O="${OUT:-/root/autodl-fs/wctr-shap-exp2-1d-gpu}"
cd "$REPO" || exit 2
export DATA_DIR="$REPO" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=26 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset PYTHONPATH PYTHONHOME
mkdir -p "$O"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$O/driver.log"; }
step() { # name command... : stop at the first failing step
  local name="$1"; shift
  "$@"; local rc=$?
  if [ "$rc" -ne 0 ]; then log "$name FAILED (exit $rc); stopped"; exit "$rc"; fi
  log "$name ok"
}

"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "no CUDA device: this job is GPU-only"; exit 2; }
sha256sum scripts/explain_exp2.py scripts/explain_exp2_s4.py > "$O/scripts.sha256"
git rev-parse HEAD > "$O/git_commit" 2>/dev/null || true

NATIVE="mamba_embeddings_seed26 mamba_onehot_seed26 transformer_embeddings_seed26 transformer_cross_attention_seed26"
HYDRA="s4_s4_embeddings_seed26 s4_s4_onehot_seed26 s4_s4_embeddings_bidir_seed26 s4_s4_onehot_bidir_seed26"

# explain_exp2.py writes lanes.csv (all kept lanes), which the S4 script reads.
# shellcheck disable=SC2086
step native "$PY" -u scripts/explain_exp2.py run --runs-dir "$PUB" --out "$O" --data-dir "$DATA_DIR" \
  --kinds flat --seeds 26 --all-lanes --device cuda --only $NATIVE
# Empty --grid-seeds: no S4ND grid arm is explained.
# shellcheck disable=SC2086
step hydra-s4 "$PY" -u scripts/explain_exp2_s4.py --runs-dir "$PUB" --out "$O" \
  --seeds 26 --grid-seeds --only $HYDRA
step summarise "$PY" scripts/explain_exp2.py summarise --out "$O"

bad=$("$PY" -c 'import json, sys
last = {}
for line in open(sys.argv[1]):
    r = json.loads(line); last[r["run"]] = r["status"]
print(" ".join(sorted(k for k, v in last.items() if v != "ok")))' "$O/checks.jsonl")
if [ -n "$bad" ]; then log "DONE, but these runs did not pass the SHAP gates: $bad"; exit 3; fi
log "ALL DONE: 8/8 runs passed both gates"
