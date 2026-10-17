#!/usr/bin/env bash
# Independent WM-as-Verifier training (action-conditioned dynamics MLP).
# Does not load or update System-1 / System-2.
#
# Usage:
#   bash scripts/run_wm_verifier_train.sh
#   MAX_STEPS=2000 bash scripts/run_wm_verifier_train.sh   # smoke
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${PYTHON:-/home/weibin02/miniconda3/envs/dial/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/weibin02/DIAL_DATASETS/LeRobot-AugPosRot-Correct}
OUT=${OUT:-/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/DIAL02/outputs/20261017/wm_verifier_posrot18}
MAX_STEPS=${MAX_STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-64}
GPU=${GPU:-0}

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$OUT"
LOG="$OUT/train.log"
echo "=== WM verifier train start $(date -Is) OUT=$OUT steps=$MAX_STEPS ===" | tee -a "$LOG"
"$PYTHON" -u scripts/train_wm_verifier.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --save-every "${SAVE_EVERY:-5000}" \
  --log-every "${LOG_EVERY:-50}" \
  >> "$LOG" 2>&1
rc=$?
echo "=== WM verifier train done rc=$rc $(date -Is) ===" | tee -a "$LOG"
exit $rc
