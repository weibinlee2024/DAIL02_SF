#!/usr/bin/env bash
# Resume preprocessing from PosttrainPnPNovelFromPlateToPlateSplitA onward.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/setup_local_paths.sh >/dev/null

export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Tasks 8-24 (skip 7 already done, start from task 8 which crashed)
REMAINING=(
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
  echo ">>> TASK=$t  ($(date))"
  echo "=============================================="
  # Clean partial output
  rm -rf "data/Replay-Correct/$t" "data/LeRobot-AugPosRot-Correct/gr1_unified.$t"
  TASK="$t" NUM_PARALLEL_JOBS="${NUM_PARALLEL_JOBS:-4}" bash scripts/preprocess_gr1_one_task.sh
  echo ">>> DONE $t"
done

echo ""
echo "ALL REMAINING TASKS DONE ($(date))"
