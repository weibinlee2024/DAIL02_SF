#!/usr/bin/env bash
# Preprocess all 24 GR1 tasks used by DIAL examples.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GR1_TASK_NAMES=(
  PnPPotatoToMicrowaveClose
  PnPMilkToMicrowaveClose
  PnPCanToDrawerClose
  PnPCupToDrawerClose
  PnPBottleToCabinetClose
  PnPWineToCabinetClose
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

for t in "${GR1_TASK_NAMES[@]}"; do
  echo "##############################"
  echo "# TASK=$t"
  echo "##############################"
  TASK="$t" bash scripts/preprocess_gr1_one_task.sh
done

echo "All tasks done. Augmented data under: $ROOT/data/LeRobot-AugPosRot-Correct/"
