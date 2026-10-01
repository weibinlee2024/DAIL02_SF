#!/bin/bash
# Offline eval on converted LIBERO LeRobot data (horizon=8 / data-config=libero).
#
# Usage:
#   bash examples/eval_loss_libero.sh <model_path> [suite] [trajs]
#
# suite: spatial|object|goal|10  (default spatial)

MODEL_PATH=${1:?"Usage: bash examples/eval_loss_libero.sh <model_path> [suite] [trajs]"}
SUITE=${2:-spatial}
TRAJS=${3:-10}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIBERO_DIR="${LIBERO_DIR:-$ROOT/data/libero}"

case "$SUITE" in
  spatial) DS="$LIBERO_DIR/libero_spatial" ;;
  object)  DS="$LIBERO_DIR/libero_object" ;;
  goal)    DS="$LIBERO_DIR/libero_goal" ;;
  10)      DS="$LIBERO_DIR/libero_10" ;;
  *) echo "Unknown suite $SUITE"; exit 1 ;;
esac

if [[ ! -d "$DS" ]]; then
  echo "Missing dataset: $DS (run preprocess_libero_dense_one_suite.sh first)" >&2
  exit 1
fi

PYTHONBREAKPOINT=0 python3 -u scripts/eval_policy_dial.py \
  --dataset-path "$DS" \
  --model_path "$MODEL_PATH" \
  --data-config libero \
  --embodiment_tag libero \
  --trajs "$TRAJS" \
  --save_results_path "$MODEL_PATH/eval_action_goal_loss_libero_${SUITE}/" \
  --plot_state \
  --vis_pca
