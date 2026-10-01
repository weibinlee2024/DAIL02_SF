#!/usr/bin/env bash
# LIBERO online eval entry (same suites as RND dense collection; horizon=8).
#
# Preferred path: reuse VLA-RFT run_libero_eval with DialPolicy backend once wired.
# This script starts the DIAL ZMQ inference server; client/env loop is left to
# VLA-RFT or a thin adapter using gr00t.eval.libero_obs_action.
#
# Usage:
#   bash examples/eval_libero.sh <model_path> [suite] [port]
#
# suite: spatial|object|goal|10

set -euo pipefail

MODEL_PATH=${1:?"Usage: bash examples/eval_libero.sh <model_path> [suite] [port]"}
SUITE=${2:-spatial}
PORT=${3:-5555}
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
DATA_CONFIG="${DATA_CONFIG:-libero}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-libero}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "$SUITE" in
  spatial) SUITE_NAME=libero_spatial ;;
  object)  SUITE_NAME=libero_object ;;
  goal)    SUITE_NAME=libero_goal ;;
  10)      SUITE_NAME=libero_10 ;;
  *) echo "Unknown suite $SUITE"; exit 1 ;;
esac

echo "=============================================="
echo "DIAL LIBERO online eval (server)"
echo "  model:      $MODEL_PATH"
echo "  suite:      $SUITE_NAME"
echo "  port:       $PORT"
echo "  H:          $N_ACTION_STEPS"
echo "  data_config:$DATA_CONFIG"
echo "=============================================="
echo "NOTE: Start VLA-RFT LIBERO client against this server, or implement"
echo "      env loop with gr00t.eval.libero_obs_action (H=8, same suites as RND)."

python3 -u scripts/inference_service_dial.py \
  --server \
  --model_path "$MODEL_PATH" \
  --embodiment_tag "$EMBODIMENT_TAG" \
  --data_config "$DATA_CONFIG" \
  --port "$PORT"
