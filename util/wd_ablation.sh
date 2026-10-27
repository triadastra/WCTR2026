#!/usr/bin/env bash
# weight_decay_exempt=false ablation on FLAT mamba.
#
# WHY THIS EXISTS
# ---------------
# Mamba-2 marks three tensors _no_weight_decay (dt_bias, A_log, D; see
# external/mamba/mamba_ssm/modules/mamba2.py:130-140). A_log is the one that
# matters: A = -exp(A_log), so decaying A_log pulls every head's state-decay
# rate toward A = -1 -- collapsing the spread of timescales the architecture
# exists to learn. config/base.yaml honours the marks by default.
#
# But the PUBLISHED runs decayed those tensors, because the exemption was not
# implemented yet. The 2026-07-29 re-run switched it on, and the split is
# stark: the 8 arms that received only the protocol change improved (mean
# -1.68%), while 3 of the 4 arms that also received the exemption got worse
# (mamba_onehot +2.65%). Two things changed at once between the published
# number and the current one, so neither can be blamed from the diagonal:
#
#                    | decay everything | exemption honoured
#   old protocol b64 | 0.04930 published| never run
#   new protocol b903| THIS SCRIPT      | 0.05061 current
#
# Filling the bottom-left cell separates protocol from optimizer. Lands near
# 0.0493 => the exemption caused the regression; near 0.0506 => the protocol
# did. This matters because the old 0.04930 still beats the corrected
# champion s4+ACA 2d at 0.04982, so the "best arm is multidimensional" claim
# may rest on this setting.
#
# Every other knob matches outputs/full/mamba_* exactly -- same data, batch
# 903, 96 steps/epoch, lr 1e-3, epochs 5000, patience 50 -- so the pair is a
# clean isolate. Sequential: one GPU, and a comparison must never straddle
# contended hardware.
#
# Usage:  bash util/wd_ablation.sh          (from the repo root, on the box)
# Resumable: any run whose output dir already holds best.pth is skipped.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
cd "$(dirname "$0")/.." || exit 1
mkdir -p outputs/wd_ablation

for seed in 26 105 115; do
  for v in embeddings onehot; do
    out="outputs/wd_ablation/mamba_${v}_seed${seed}"
    # train.py writes best.pth, NOT metrics.json -- that comes from
    # inference.py afterwards. Gate on best.pth or every run re-trains.
    if [ -f "$out/best.pth" ]; then echo "[skip] $out"; continue; fi
    echo "[run] $out  $(date +%H:%M:%S)"
    python scripts/train.py --model mamba --variant "$v" --seed "$seed" \
      --no-weight-decay-exempt --data-dir /root --out-dir "$out" \
      > "$out.log" 2>&1
    rc="${PIPESTATUS[0]}"
    if [ "$rc" -eq 0 ]; then echo "  done"; else echo "  [FAIL] rc=$rc"; fi
  done
done

# Test metrics require a second pass; training alone never scores the test split.
for seed in 26 105 115; do
  for v in embeddings onehot; do
    out="outputs/wd_ablation/mamba_${v}_seed${seed}"
    [ -f "$out/best.pth" ] || continue
    [ -f "$out/metrics.json" ] && { echo "[skip-infer] $out"; continue; }
    echo "[infer] $out"
    # --normalized is NOT optional here. Without it inference scores in real
    # dollars (mode "one_step", value_mse ~3e14) and never writes
    # norm_mse_combined at all, so the run cannot be compared to anything in
    # outputs/full. This invocation mirrors run_full_test.sh:40-43 exactly.
    python scripts/inference.py --model mamba --variant "$v" \
      --checkpoint "$out/best.pth" --seed "$seed" --device cuda \
      --data-dir /root --out-dir "$out" --normalized \
      >> "$out.log" 2>&1
    echo "  rc=${PIPESTATUS[0]}"
  done
done
echo "ALL_DONE $(date +%H:%M:%S)"
