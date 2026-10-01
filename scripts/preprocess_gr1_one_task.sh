#!/usr/bin/env bash
# Preprocess one GR1 task: HDF5 replay -> EEF poses -> AugPosRot LeRobot dataset.
# Requires: robosuite v1.5.1 + robocasa-gr1-tabletop-tasks + pin/mink (see README).
#
# Usage:
#   TASK=PnPCupToDrawerClose bash scripts/preprocess_gr1_one_task.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/setup_local_paths.sh >/dev/null

TASK="${TASK:?Set TASK=PnPCupToDrawerClose (or another GR1 task name)}"
RAW="${RAW_ROOT:-$ROOT/data/raw/PhysicalAI-Robotics-GR00T-Teleop-Sim}"
HDF5="$RAW/HDF5/${TASK}.hdf5"
LEROBOT_IN="$RAW/LeRobot/gr1_unified.${TASK}"
REPLAY_OUT="$ROOT/data/Replay-Correct/${TASK}"
LEROBOT_OUT="$ROOT/data/LeRobot-AugPosRot-Correct/gr1_unified.${TASK}"

if [[ ! -f "$HDF5" ]]; then
  echo "Missing HDF5: $HDF5" >&2
  echo "Run: bash scripts/download_gr1_raw.sh" >&2
  exit 1
fi
if [[ ! -d "$LEROBOT_IN" ]]; then
  echo "Missing LeRobot: $LEROBOT_IN" >&2
  exit 1
fi

mkdir -p "$(dirname "$REPLAY_OUT")" "$(dirname "$LEROBOT_OUT")"

echo "=== Step1: extract EEF poses for $TASK ==="
python preprocessing/extract_and_visualize_3d-pos_6d-rot_from_gr1.py \
  --dataset "$HDF5" \
  --output_dir "$REPLAY_OUT" \
  --render_image_names egoview \
  --verbose \
  --render_height 800 \
  --render_width 1280 \
  --num_parallel_jobs "${NUM_PARALLEL_JOBS:-4}"

echo "=== Step2: augment LeRobot with poses ==="
python preprocessing/aug_lerobot_data.py \
  --lerobot_base_path "$LEROBOT_IN" \
  --replay_base_path "$REPLAY_OUT/parquet/" \
  --output_base_path "$LEROBOT_OUT"

echo "Wrote $LEROBOT_OUT"
