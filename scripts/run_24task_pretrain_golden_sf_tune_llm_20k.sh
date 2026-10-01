#!/usr/bin/env bash
# 24-task DIAL golden pretrain: 20k steps, save every 5k.
# --tune-llm + state-future. Mixed precision = Trainer AMP (fp32 master, bf16 compute).
# Do NOT set MODEL_BF16. bs=2 accum=32 → effective batch 64.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-"/media/weibin/weibin01/DIAL_DATASETS/LeRobot-AugPosRot-Correct"}
PYTHON=${PYTHON:-/home/weibin/anaconda3/envs/dial/bin/python}
OUT=${OUT:-"/media/weibin/weibin02/DIAL/20260918/pretrain-golden-sf-24task-20k-tune-llm"}
LOG=${LOG:-"/media/weibin/weibin02/DIAL/20260918/pretrain_24task_sf_tune_llm.log"}
MAX_STEPS=${MAX_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-32}
NUM_GPUS=${NUM_GPUS:-1}
BASE_CFG=${BASE_CFG:-"$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json"}
mkdir -p "$OUT" "$(dirname "$LOG")"

TASKS=(
  PnPBottleToCabinetClose
  PnPCanToDrawerClose
  PnPCupToDrawerClose
  PnPMilkToMicrowaveClose
  PnPPotatoToMicrowaveClose
  PnPWineToCabinetClose
  PosttrainPnPNovelFromCuttingboardToBasketSplitA
  PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
  PosttrainPnPNovelFromCuttingboardToPanSplitA
  PosttrainPnPNovelFromCuttingboardToPotSplitA
  PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA
  PosttrainPnPNovelFromPlacematToBasketSplitA
  PosttrainPnPNovelFromPlacematToBowlSplitA
  PosttrainPnPNovelFromPlacematToPlateSplitA
  PosttrainPnPNovelFromPlacematToTieredshelfSplitA
  PosttrainPnPNovelFromPlateToBowlSplitA
  PosttrainPnPNovelFromPlateToCardboardboxSplitA
  PosttrainPnPNovelFromPlateToPanSplitA
  PosttrainPnPNovelFromPlateToPlateSplitA
  PosttrainPnPNovelFromTrayToCardboardboxSplitA
  PosttrainPnPNovelFromTrayToPlateSplitA
  PosttrainPnPNovelFromTrayToPotSplitA
  PosttrainPnPNovelFromTrayToTieredbasketSplitA
  PosttrainPnPNovelFromTrayToTieredshelfSplitA
)

DS=()
for t in "${TASKS[@]}"; do
  p="$DATA_ROOT/gr1_unified.$t"
  if [[ ! -d "$p" ]]; then
    echo "ERROR: missing dataset $p" | tee -a "$LOG"
    exit 1
  fi
  DS+=("$p")
done

if [[ ! -f "$BASE_CFG" ]]; then
  echo "ERROR: missing BASE_CFG=$BASE_CFG" | tee -a "$LOG"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset MODEL_BF16
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 24task SF pretrain (tune-llm, AMP bf16) start $(date -Is) ===" | tee -a "$LOG"
echo "tasks=${#TASKS[@]} BASE_CFG=$BASE_CFG OUT=$OUT MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS eff_batch=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))" | tee -a "$LOG"
echo "FLAGS: --tune-llm --use-state-future --compute-state-future-loss MODEL_BF16=unset AMP=bf16" | tee -a "$LOG"

"$PYTHON" -u scripts/dual_system_train.py \
  --bridge_type golden \
  --dataset-path "${DS[@]}" \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 \
  --data_split "[:-10]" \
  --num-gpus "$NUM_GPUS" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM" \
  --dataloader-num-workers 2 --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" \
  --base_model_path "$BASE_CFG" \
  --compute-bridge-loss --tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard \
  --use-state-future --compute-state-future-loss --state-future-loss-weight 0.5 \
  --output_dir "$OUT" >> "$LOG" 2>&1
rc=$?
echo "24TASK-PRETRAIN-TUNE-LLM DONE rc=$rc $(date -Is)" >> "$LOG"
exit $rc
