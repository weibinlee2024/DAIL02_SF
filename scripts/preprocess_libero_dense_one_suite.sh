#!/usr/bin/env bash
# Convert one LIBERO suite of RND dense NPZ -> DIAL LeRobot v2.0.
#
# Usage:
#   SUITE=spatial MAX_EPISODES=20 bash scripts/preprocess_libero_dense_one_suite.sh
#   SUITE=spatial bash scripts/preprocess_libero_dense_one_suite.sh
#
# Env:
#   NPZ_ROOT   default: VLA-RFT rollouts/rl_dense_npz
#   OUT_ROOT   default: $ROOT/data/libero
#   TAG        default: rnd_dense_all_v2
#   MAX_EPISODES  optional smoke cap
#   SUCCESS_ONLY  if 1, drop failures (ablation only)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SUITE="${SUITE:?Set SUITE=spatial|object|goal|10}"
TAG="${TAG:-rnd_dense_all_v2}"
NPZ_ROOT="${NPZ_ROOT:-/home/weibin/projects3/Model-based Model_Code/VLA-RFT/rollouts/rl_dense_npz}"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/libero}"

case "$SUITE" in
  spatial) OUT_NAME=libero_spatial ;;
  object)  OUT_NAME=libero_object ;;
  goal)    OUT_NAME=libero_goal ;;
  10|libero_10) OUT_NAME=libero_10; SUITE=10 ;;
  *) echo "Unknown SUITE=$SUITE"; exit 1 ;;
esac

OUT_DIR="$OUT_ROOT/$OUT_NAME"
mkdir -p "$OUT_DIR"

EXTRA=()
if [[ -n "${MAX_EPISODES:-}" ]]; then
  EXTRA+=(--max_episodes "$MAX_EPISODES")
fi
if [[ "${SUCCESS_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--success_only)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA+=(--overwrite)
fi

echo "=== LIBERO dense NPZ -> LeRobot ==="
echo "  suite=$SUITE tag=$TAG"
echo "  npz=$NPZ_ROOT"
echo "  out=$OUT_DIR"

python3 -u preprocessing/convert_libero_dense_npz_to_lerobot.py \
  --npz_root "$NPZ_ROOT" \
  --output_dir "$OUT_DIR" \
  --suites "$SUITE" \
  --tag "$TAG" \
  "${EXTRA[@]}"

python3 -u preprocessing/validate_libero_lerobot.py --dataset_path "$OUT_DIR"
echo "Wrote $OUT_DIR"
