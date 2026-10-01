#!/usr/bin/env bash
# Preprocess remaining 18 tasks (skip the 6 PnPClose already done, redo the empty Posttrain one).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/setup_local_paths.sh >/dev/null

export MUJOCO_GL="${MUJOCO_GL:-egl}"

REMAINING=(
  PosttrainPnPNovelFromPlacematToBowlSplitA
  PosttrainPnPNovelFromPlateToPlateSplitA
  PosttrainPnPNovelFromPlacematToPlateSplitA
  PosttrainPnPNovelFromCuttingboardToPotSplitA
  PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
  PosttrainPnPNovelFromCuttingboardToPanSplitA
  PosttrainPnPNovelFromTrayToCardboardboxSplitA
  PosttrainPnPNovelFromTrayToTieredshelfSplitA
  PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA
  PosttrainPnPNovelFromPlacematToTieredshelfSplitA
  PosttrainPnPNovelFromPlateToCardboardboxSplitA
  PosttrainPnPNovelFromPlacematToBasketSplitA
  PosttrainPnPNovelFromPlateToPanSplitA
  PosttrainPnPNovelFromTrayToTieredbasketSplitA
  PosttrainPnPNovelFromTrayToPotSplitA
  PosttrainPnPNovelFromPlateToBowlSplitA
  PosttrainPnPNovelFromCuttingboardToBasketSplitA
  PosttrainPnPNovelFromTrayToPlateSplitA
)

for t in "${REMAINING[@]}"; do
  echo ""
  echo "=============================================="
  echo ">>> TASK=$t"
  echo "=============================================="
  # Clean any partial output so replay/aug start fresh for this task
  rm -rf "data/Replay-Correct/$t" "data/LeRobot-AugPosRot-Correct/gr1_unified.$t"
  TASK="$t" NUM_PARALLEL_JOBS="${NUM_PARALLEL_JOBS:-8}" bash scripts/preprocess_gr1_one_task.sh
  echo ">>> DONE $t"
done

echo ""
echo "ALL REMAINING TASKS DONE"
