#!/usr/bin/env bash
# Formal LIBERO-plus fine-tune from the Robocasa 20k checkpoint.
#
# Task-level 8/2 split: task_index 0..7 train, 8..9 held-out (per suite).
# Config: tune_llm (36 layers), UNIFIED_TAG=gr1, two cameras, AV1 via PyAV.
# SF and bridge loss stay on, matching the 36-layer checkpoint:
#   mean_pool SF, dim 128, horizon 16, loss weight 0.5
#   (LIBERO future EEF is 8-d, zero-padded to 128, pad masked out)
#   bridge loss on the main-camera frame at t+16, separate loss projector.
#
# Usage:
#   bash scripts/run_libero_plus_ft_20k_unified.sh
#   MAX_STEPS=2000 SAVE_STEPS=500 OUT=/media/.../run2 bash scripts/run_libero_plus_ft_20k_unified.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/weibin02/miniconda3/envs/dial/bin/python}"
BASE_CKPT="${BASE_CKPT:-/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/DIAL02/outputs/20261102/checkpoint_copies/policy_meal_sf36L_e2e_ckpt20000_meanpool_infer}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/Libero_plus/lerobot/root/.cache/huggingface/lerobot/Sylvest}"

MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
WORKERS="${WORKERS:-2}"
DATA_SPLIT="${DATA_SPLIT:-[:]}"
TASK_INDICES="${TASK_INDICES:-0 1 2 3 4 5 6 7}"
SELECT_LAYER="${SELECT_LAYER:-36}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"
UNIFIED_TAG="${UNIFIED_TAG:-gr1}"
OUT="${OUT:-/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/DIAL02/outputs/20261112/libero_plus_ft20k_unified}"
LOG="${LOG:-$OUT/train.log}"

SUITES=(
  "$LIBERO_PLUS_ROOT/libero_plus_spatial"
  "$LIBERO_PLUS_ROOT/libero_plus_object"
  "$LIBERO_PLUS_ROOT/libero_plus_goal"
  "$LIBERO_PLUS_ROOT/libero_plus_10"
)
for suite in "${SUITES[@]}"; do
  [[ -d "$suite" ]] || { echo "ERROR: missing suite $suite" >&2; exit 1; }
done
[[ -d "$BASE_CKPT" ]] || { echo "ERROR: missing BASE_CKPT=$BASE_CKPT" >&2; exit 1; }

mkdir -p "$OUT" "$(dirname "$LOG")"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Single-GPU full tune-llm needs bf16 weights (the 20k recipe used AMP +
# DeepSpeed ZeRO-2 on 2 GPUs). Set MODEL_BF16=0 only for a ZeRO/multi-GPU run.
MODEL_BF16="${MODEL_BF16:-1}"
if [[ "$MODEL_BF16" == "1" ]]; then
  export MODEL_BF16=1
else
  unset MODEL_BF16
fi

echo "=== LIBERO-plus formal FT ==="
echo "  BASE_CKPT:    $BASE_CKPT"
echo "  suites:       ${#SUITES[@]} (task_indices $TASK_INDICES train; 8,9 held-out)"
echo "  batch:        $BATCH_SIZE x accum $GRAD_ACCUM x gpus $NUM_GPUS"
echo "  steps:        $MAX_STEPS (save every $SAVE_STEPS)"
echo "  select_layer: $SELECT_LAYER; tune_llm=1; unified=$UNIFIED_TAG; MODEL_BF16=$MODEL_BF16"
echo "  video:        $VIDEO_BACKEND; split=$DATA_SPLIT"
echo "  output:       $OUT"
echo "  log:          $LOG"

"$PYTHON" -u scripts/dual_system_train.py \
  --bridge_type end2end \
  --dataset-path "${SUITES[@]}" \
  --data-config libero libero libero libero \
  --embodiment_tag libero libero libero libero \
  --unified_embodiment_tag "$UNIFIED_TAG" \
  --task_indices $TASK_INDICES \
  --data_split "$DATA_SPLIT" \
  --num-gpus "$NUM_GPUS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --dataloader-num-workers "$WORKERS" \
  --video_backend "$VIDEO_BACKEND" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --base_model_path "$BASE_CKPT" \
  --compute-bridge-loss \
  --use-separate-projector-for-loss \
  --use-state-future \
  --compute-state-future-loss \
  --state-future-decoder mean_pool \
  --state-future-dim 128 \
  --state-future-horizon 16 \
  --state-future-loss-weight 0.5 \
  --tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --no-tune-bridge-visual \
  --no-tune-bridge-goal \
  --tune-bridge-embedding \
  --select_layer "$SELECT_LAYER" \
  --use_image_type_embedding \
  --ignore_lang_prefix \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay 1e-5 \
  --warmup_ratio 0.05 \
  --report_to tensorboard \
  --output_dir "$OUT" 2>&1 | tee "$LOG"

echo "=== formal FT done: $OUT ==="
