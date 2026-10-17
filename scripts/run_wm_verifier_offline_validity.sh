#!/usr/bin/env bash
# (4) Offline score validity — must pass before (5) online K-sweep.
#
# Usage:
#   POLICY_PATH=<ckpt> WM_CKPT=<wm.pt> bash scripts/run_wm_verifier_offline_validity.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${PYTHON:-/home/weibin02/miniconda3/envs/dial/bin/python}
POLICY_PATH=${POLICY_PATH:?"set POLICY_PATH to frozen policy checkpoint"}
WM_CKPT=${WM_CKPT:?"set WM_CKPT to wm_verifier checkpoint-*.pt"}
OUT_JSON=${OUT_JSON:-"$(dirname "$WM_CKPT")/offline_score_validity.json"}
GPU=${GPU:-0}
TRAJS=${TRAJS:-4}
POINTS=${POINTS:-4}

export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== WM offline validity K=4,8 start $(date -Is) ==="
"$PYTHON" -u scripts/eval_wm_verifier_offline.py \
  --policy-path "$POLICY_PATH" \
  --wm-ckpt "$WM_CKPT" \
  --ks 4 8 \
  --trajs "$TRAJS" \
  --points "$POINTS" \
  --save-json "$OUT_JSON"
echo "=== done $(date -Is) json=$OUT_JSON ==="
echo "NOTE: (5) online K-sweep (K=1/4/8) only after offline Δmse / pairwise / Spearman look OK."
