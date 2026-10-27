#!/usr/bin/env bash
# Exp 1 (aggregated series) at many seeds, for all 16 STOCHASTIC arms.
#
# WHY
# ---
# Exp 1's neural arms have a median coefficient of variation of 37% across the
# 3 protocol seeds; Exp 2's is 0.6%. Sixty times worse, and one arm
# (transformer+onehot) has a standard deviation as large as its mean. Concretely:
#
#     lstm+embeddings   seed 26 = 0.00350   seed 105 = 0.03408   seed 115 = 0.03466
#
# Reporting 0.02408 as that arm's score is close to meaningless.
#
# This is structural, not a bug. Exp 1 sums all 903 lanes into ONE series:
# 132 training samples and 5 test points, against Exp 2's 119,196 and 4,515.
# A ~500K-parameter model fitting 132 points lands in a very different minimum
# per initialisation, and 5 test months cannot average that away.
#
# So re-running the same 3 seeds achieves nothing -- three more draws from the
# same wide distribution. More seeds do not reduce the variance either; they pin
# down the MEAN. Standard error falls as std/sqrt(n):
#
#     n=3  -> SE ~ 21% of the mean      n=10 -> ~12%      n=25 -> ~7%
#
# IS 25 SEEDS UNFAIR TO EXP 2, WHICH HAS 3?
# ------------------------------------------
# No, with one caveat that matters.
#
# For the mean-of-scores statistic (what Exp2_PerCombo reports),
# E[mean of n scores] = E[score] for ANY n. More seeds shrink the standard error
# of the estimate, not the estimate. So this does not flatter Exp 1's models --
# it measures them more precisely. Fairness holds WITHIN each experiment: every
# Exp 1 arm gets the same n, every Exp 2 arm gets the same n. The two are never
# compared to each other; different data, different scales (0.003-0.035 vs 0.05).
#
# THE CAVEAT: this reasoning does NOT extend to the ENSEMBLE statistic, which
# genuinely improves with n -- mean(MSE) - MSE(mean) = Var(predictions), so
# averaging more seeds cancels more variance. An Exp1 ensemble at 25 seeds must
# never be placed beside Exp2_Ensemble at 3. Do not build one from this output.
#
# WHY ONLY 16 ARMS
# ----------------
# The three naive baselines are deterministic and keep their existing rows.
# XGBoost and LightGBM are not: their row/feature subsampling now receives the
# run seed, so both variants must get the same 25-seed coverage as the neural
# and SSM arms.
#
# Also fixes a stray: the published Exp 1 used seed 42 for the three naive
# baselines while claiming 26/105/115. Harmless numerically (they are
# deterministic) but the sheet misreports its own protocol.
#
# Usage:   bash util/run_exp1_seeds.sh              # 25 seeds
#          SEEDS="26 105 115" bash util/run_exp1_seeds.sh   # protocol only
# Resumable: only a provenance-validated, identity-matched artifact is skipped.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base

DATA_DIR="${DATA_DIR:-/root}"
# 25 seeds: the 3 protocol seeds (26/105/115, drawn from 1-1000 per the
# paper's original protocol) + 22 additional seeds. The 22 are a FIXED,
# hand-spaced list over [7, 226] (increments 3-15, chosen to be distinct from
# each other and from the protocol seeds) -- NOT a seeded random draw; an
# earlier version of this comment claimed randint(1,1000) under meta-seed 42,
# which does not reproduce this list (checked 2026-08-03). The list itself is
# the specification: committed before any run, every run reported, none
# discarded. Seeds are arbitrary RNG labels, so the variance estimate is
# unaffected; describe provenance in the paper as "a fixed pre-registered
# list", not as a range draw.
SEEDS="${SEEDS:-26 105 115 7 13 19 31 44 58 67 73 88 91 104 119 127 138 152 166 173 188 194 205 211 226}"
# Stochastic arms only -- see WHY ONLY 16 ARMS.
MODELS="${MODELS:-lstm gru transformer gpt mamba s4 xgboost lightgbm}"
VARIANTS="${VARIANTS:-embeddings onehot}"
PY="${PY:-python}"

mkdir -p outputs/agg
total=0; trained=0; skipped=0; failed=0
for s in $SEEDS; do for m in $MODELS; do for v in $VARIANTS; do
  total=$((total+1))
  out="outputs/agg/${m}_${v}_seed${s}"
  if $PY scripts/run_is_current.py "$out/metrics.json" --aggregate \
      --input-len 36 --lag-count 12 --expected-seed "$s" \
      --expected-model "$m" --expected-variant "$v"; then
    skipped=$((skipped+1)); continue
  fi
  mkdir -p "$out"
  echo "[exp1] $out  $(date -u +%FT%TZ)"
  # --aggregate is what makes this Exp 1. config/exp1.yaml (loaded automatically
  # alongside it) pins batch 64: base.yaml's 903 exceeds the entire 132-sample
  # dataset and with drop_last=True yields ZERO batches.
  $PY scripts/train.py --model "$m" --variant "$v" --seed "$s" --aggregate \
      --data-dir "$DATA_DIR" --out-dir "$out" > "$out/train.log" 2>&1
  if [ $? -ne 0 ]; then echo "  [FAIL] train"; failed=$((failed+1)); continue; fi
  # --normalized is mandatory: without it inference scores in real dollars and
  # never writes norm_mse_combined, so the run cannot join the table at all.
  # --aggregate must match training or the loader shape differs.
  $PY scripts/inference.py --model "$m" --variant "$v" --seed "$s" --aggregate \
      --checkpoint "$out/best.pth" --device cuda --data-dir "$DATA_DIR" \
      --out-dir "$out" --normalized > "$out/inference.log" 2>&1
  if [ $? -ne 0 ]; then echo "  [FAIL] infer"; failed=$((failed+1)); continue; fi
  trained=$((trained+1))
  echo "  ok ($trained trained / $skipped skipped / $failed failed of $total)"
done; done; done

echo "EXP1_DONE $(date -u +%FT%TZ) | trained=$trained skipped=$skipped failed=$failed total=$total"
echo "Next: rebuild on the box (outputs/agg is box-only):"
echo "  python scripts/build_paper_results.py"
