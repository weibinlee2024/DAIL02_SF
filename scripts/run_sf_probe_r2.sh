#!/usr/bin/env bash
# Run linear probe R² on a checkpoint (teacher: representation verifiability).
# Usage:
#   CKPT=.../checkpoint-10000 bash scripts/run_sf_probe_r2.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${PYTHON:-/home/weibin/anaconda3/envs/dial/bin/python}
CKPT=${CKPT:?"set CKPT=/path/to/checkpoint-XXXX"}
DATA_ROOT=${DATA_ROOT:-"/media/weibin/weibin02/LeRobot-AugPosRot-Correct"}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop}
OUT=${OUT:-"$CKPT/sf_probe_r2"}
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/home/weibin/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== probe R² CKPT=$CKPT OUT=$OUT $(date -Is) ==="
"$PYTHON" -u scripts/diag_sf_per_frame_and_probe.py \
  --model-path "$CKPT" \
  --data-root "$DATA_ROOT" \
  --data-config "$DATA_CONFIG" \
  --out-dir "$OUT"
echo "DONE $(date -Is) → $OUT/sf_per_frame_and_probe.json"
