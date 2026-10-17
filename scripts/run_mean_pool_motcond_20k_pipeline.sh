#!/usr/bin/env bash
# mean_pool + shat + w=0.05 + Dense cond-experts + detach ON.
#   golden 20k → end2end 20k (teacher mainline). D_sf=30, 17 SF tokens.
#   SF tokens use the SF K/V expert; visual tokens use the vis expert.
#   Do not combine with --use-mot-system1. mean_pool has no mid_hidden → shat only.
#
# Usage:
#   bash scripts/run_mean_pool_motcond_20k_pipeline.sh
#   STAGE=pre|ft|all bash scripts/run_mean_pool_motcond_20k_pipeline.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-/home/weibin02/DIAL_DATASETS/LeRobot-AugPosRot-Correct}
PYTHON=${PYTHON:-/home/weibin02/miniconda3/envs/dial/bin/python}
W=${SF_DUAL_W:-0.05}
WTAG=$(echo "$W" | tr '.' 'p')
OUT_ROOT=${ALIGN_OUT_ROOT:-/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/DIAL02/outputs/20261017}
STAGE=${STAGE:-all}
SELECT_LAYER=${SELECT_LAYER:-12}

if [[ -n "${SMOKE_STEPS:-}" ]]; then
  PRE_STEPS=${SMOKE_STEPS}
  PRE_SAVE=${SMOKE_STEPS}
else
  PRE_STEPS=${PRE_STEPS:-20000}
  PRE_SAVE=${PRE_SAVE:-5000}
fi
# Teacher mainline: golden 20k + end2end 20k.
FT_STEPS=${FT_STEPS:-20000}
FT_SAVE=${FT_SAVE:-5000}

# This machine: 2×4090. Effective batch = bs × accum × gpus = 4×8×2 = 64.
# Override with NUM_GPUS=1 GRAD_ACCUM=16 if you want single-GPU.
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
NUM_GPUS=${NUM_GPUS:-2}
TRAJS=${TRAJS:-10}
BASE_CFG=${BASE_CFG:-"$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json"}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop}

PRE_OUT="${OUT_ROOT}/mean_pool_motcond_w${WTAG}_tuneLLM_L${SELECT_LAYER}_pre_${PRE_STEPS}"
FT_OUT="${OUT_ROOT}/mean_pool_motcond_w${WTAG}_tuneLLM_L${SELECT_LAYER}_ft_${FT_STEPS}"
PRE_CKPT="${PRE_OUT}/checkpoint-${PRE_STEPS}"
PRE_LOG="${PRE_OUT}.log"
FT_LOG="${FT_OUT}.log"

export MODEL_BF16=${MODEL_BF16:-0}
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SF_FORCE_KEY_DIM=30
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/home/weibin02/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DS_CONFIG="${DS_CONFIG:-$ROOT/configs/ds_zero2_bf16.json}"
export TOKENIZERS_PARALLELISM=false
export STAGGER_INIT_SECONDS="${STAGGER_INIT_SECONDS:-180}"

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
  [[ -d "$p" ]] || { echo "ERROR: missing $p"; exit 1; }
  DS+=("$p")
done
[[ -f "$BASE_CFG" ]] || { echo "ERROR: missing BASE_CFG=$BASE_CFG"; exit 1; }

COMMON=(
  --data-config "$DATA_CONFIG"
  --embodiment_tag gr1
  --data_split "[:-10]"
  --num-gpus "$NUM_GPUS" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM"
  --dataloader-num-workers 2
  --compute-bridge-loss --tune-llm --no-tune-visual --tune-projector --tune-diffusion-model
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding
  --select_layer "$SELECT_LAYER" --use_image_type_embedding --ignore_lang_prefix
  --report_to tensorboard
  --use-state-future --compute-state-future-loss --state-future-loss-weight "$W"
  --state-future-decoder mean_pool --state-future-dim 30
  --loss-combine-mode additive
  --feed-state-future-to-dit
  --sf-dit-repr shat
  --sf-tokens-detach
  --use-mot-cond-experts --mot-include-sf --mot-sf-num-tokens 17
)

run_pre() {
  if [[ -d "$PRE_CKPT" ]]; then
    echo "SKIP pre: $PRE_CKPT exists"
    return 0
  fi
  mkdir -p "$PRE_OUT"
  echo "=== mean_pool+motcond PRE golden steps=$PRE_STEPS w=$W dim=30 layer=$SELECT_LAYER start $(date -Is) OUT=$PRE_OUT ===" | tee -a "$PRE_LOG"
  "$PYTHON" -u scripts/dual_system_train.py \
    --bridge_type golden \
    --dataset-path "${DS[@]}" \
    --max-steps "$PRE_STEPS" --save-steps "$PRE_SAVE" \
    --base_model_path "$BASE_CFG" \
    "${COMMON[@]}" \
    --output_dir "$PRE_OUT" >> "$PRE_LOG" 2>&1
  local rc=$?
  echo "TRAIN pre rc=$rc $(date -Is)" | tee -a "$PRE_LOG"
  return $rc
}

run_ft() {
  if [[ -n "${SMOKE_STEPS:-}" ]]; then
    echo "SMOKE set: skip finetune"
    return 0
  fi
  local base_ckpt=${BASE_CKPT:-"$PRE_CKPT"}
  [[ -d "$base_ckpt" ]] || { echo "ERROR: missing BASE_CKPT=$base_ckpt"; exit 1; }
  if [[ -d "$FT_OUT/checkpoint-${FT_STEPS}" ]]; then
    echo "SKIP ft: $FT_OUT/checkpoint-${FT_STEPS} exists"
    return 0
  fi
  mkdir -p "$FT_OUT"
  echo "=== mean_pool+motcond FT end2end steps=$FT_STEPS base=$base_ckpt start $(date -Is) OUT=$FT_OUT ===" | tee -a "$FT_LOG"
  "$PYTHON" -u scripts/dual_system_train.py \
    --bridge_type end2end \
    --dataset-path "${DS[@]}" \
    --max-steps "$FT_STEPS" --save-steps "$FT_SAVE" \
    --base_model_path "$base_ckpt" \
    "${COMMON[@]}" \
    --output_dir "$FT_OUT" >> "$FT_LOG" 2>&1
  local rc=$?
  echo "TRAIN ft rc=$rc $(date -Is)" | tee -a "$FT_LOG"
  return $rc
}

cd "$ROOT"
mkdir -p "$OUT_ROOT"
case "$STAGE" in
  pre) run_pre || exit $? ;;
  ft)  run_ft || exit $? ;;
  all)
    run_pre || exit $?
    run_ft || exit $?
    ;;
  *) echo "STAGE=$STAGE invalid (pre|ft|all)"; exit 1 ;;
esac
echo "=== mean_pool+motcond pipeline done $(date -Is) ==="
