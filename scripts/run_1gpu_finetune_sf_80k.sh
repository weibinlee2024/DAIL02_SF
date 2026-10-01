#!/usr/bin/env bash
# SF finetune 80k (end2end) from pretrain-golden-sf-80k/checkpoint-80000.
# Unfreezes LLM (--tune-llm); visual still frozen. Independent output dir; ckpt every 10k, no deletion.
set -u
ROOT=/home/weibin/DIAL-master
DATA_ROOT=$ROOT/Datasets/LeRobot-AugPosRot-Correct
PYTHON=/home/weibin/miniconda3/envs/dial/bin/python
OUT=/media/weibin/weibin02/DIAL/20260918/finetune-end2end-sf-80k-tune-llm
LOG=/media/weibin/weibin02/DIAL/20260918/finetune_sf_80k_tune_llm.log
mkdir -p "$OUT"

TASKS="PnPBottleToCabinetClose PnPCanToDrawerClose PnPCupToDrawerClose PnPMilkToMicrowaveClose PnPPotatoToMicrowaveClose PnPWineToCabinetClose PosttrainPnPNovelFromCuttingboardToBasketSplitA PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA PosttrainPnPNovelFromCuttingboardToPanSplitA PosttrainPnPNovelFromCuttingboardToPotSplitA PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA PosttrainPnPNovelFromPlacematToBasketSplitA PosttrainPnPNovelFromPlacematToBowlSplitA PosttrainPnPNovelFromPlacematToPlateSplitA PosttrainPnPNovelFromPlacematToTieredshelfSplitA PosttrainPnPNovelFromPlateToBowlSplitA PosttrainPnPNovelFromPlateToCardboardboxSplitA PosttrainPnPNovelFromPlateToPanSplitA PosttrainPnPNovelFromPlateToPlateSplitA PosttrainPnPNovelFromTrayToCardboardboxSplitA PosttrainPnPNovelFromTrayToPlateSplitA PosttrainPnPNovelFromTrayToPotSplitA PosttrainPnPNovelFromTrayToTieredbasketSplitA PosttrainPnPNovelFromTrayToTieredshelfSplitA"

DS=()
for t in $TASKS; do DS+=("$DATA_ROOT/gr1_unified.$t"); done

cd "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Mixed precision via Trainer AMP (bf16=True). Do NOT cast weights.
unset MODEL_BF16
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u scripts/dual_system_train.py \
  --bridge_type end2end \
  --dataset-path "${DS[@]}" \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 \
  --data_split "[:-10]" \
  --num-gpus 1 --batch-size 2 --gradient-accumulation-steps 32 \
  --dataloader-num-workers 2 --max-steps 80000 --save-steps 10000 \
  --base_model_path "$ROOT/outputs/pretrain-golden-sf-80k/checkpoint-80000" \
  --compute-bridge-loss --tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --use-state-future --compute-state-future-loss --state-future-loss-weight 0.5 \
  --report_to tensorboard \
  --output_dir "$OUT" > "$LOG" 2>&1
echo "SF-FINETUNE-80K-TUNE-LLM DONE rc=$?" >> "$LOG"
