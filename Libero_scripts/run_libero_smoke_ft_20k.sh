#!/usr/bin/env bash
# LIBERO smoke fine-tune from a Robocasa 20k checkpoint.
#
# Verifies that:
#   1) LibreDataConfig can read the converted LIBERO LeRobot dataset;
#   2) the 20k checkpoint loads with max_state_dim/max_action_dim=128;
#   3) one or a few training steps run and write experiment_cfg/metadata.json.
#
# Usage:
#   BASE_CKPT=/path/to/policy_..._ckpt20000_meanpool_infer \
#     bash scripts/run_libero_smoke_ft_20k.sh
#
# Optional:
#   SUITE=libero_spatial LIBERO_DIR=/path/to/data/libero \
#   MAX_STEPS=20 SAVE_STEPS=20 BATCH_SIZE=1 GRAD_ACCUM=1 DATA_SPLIT=[:20] \
#   UNIFIED_TAG=gr1 CUDA_VISIBLE_DEVICES=1 bash scripts/run_libero_smoke_ft_20k.sh
#
# tune_llm is ON by default (matches the 20k recipe). Set TUNE_LLM=0 only
# when you intentionally want a frozen-LLM baseline.
#
# LIBERO-plus videos are AV1; set VIDEO_BACKEND=torchvision_av for those
# suites (decord cannot decode AV1). Original LIBERO H.264 works with decord.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/weibin02/miniconda3/envs/dial/bin/python}"
LIBERO_DIR="${LIBERO_DIR:-/home/weibin02/DIAL02_20261017/data/libero}"
BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to the Robocasa 20k checkpoint or its inference copy}"
SUITE="${SUITE:-libero_spatial}"
DATASET_PATH="${DATASET_PATH:-$LIBERO_DIR/$SUITE}"

MAX_STEPS="${MAX_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
WORKERS="${WORKERS:-0}"
DATA_SPLIT="${DATA_SPLIT:-[:20]}"
SELECT_LAYER="${SELECT_LAYER:-36}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
# LIBERO original videos are H.264 (decord works). Official LIBERO-plus
# LeRobot videos are AV1; decord fails, use torchvision_av (PyAV).
VIDEO_BACKEND="${VIDEO_BACKEND:-decord}"
# Match the Robocasa 20k recipe: the LLM is trainable by default.
# Set TUNE_LLM=0 only for a memory-constrained smoke.
TUNE_LLM="${TUNE_LLM:-1}"
# Optional: share the trained GR1 encoder/decoder weights with LIBERO
# (e.g. UNIFIED_TAG=gr1). Empty = train the new libero category weights.
UNIFIED_TAG="${UNIFIED_TAG:-}"

OUT="${OUT:-$ROOT/outputs/libero-smoke-ft-20k/$SUITE}"
LOG="${LOG:-$OUT/smoke.log}"
mkdir -p "$OUT" "$(dirname "$LOG")"

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "ERROR: LIBERO dataset not found: $DATASET_PATH" >&2
  exit 1
fi
if [[ ! -d "$BASE_CKPT" ]]; then
  echo "ERROR: BASE_CKPT not found: $BASE_CKPT" >&2
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep fp32 master weights by default (AMP bf16 compute).
# Set MODEL_BF16=1 to cast weights/optimizer states to bf16 on a busy GPU.
if [[ "${MODEL_BF16:-0}" != "1" ]]; then
  unset MODEL_BF16
fi

echo "=== LIBERO smoke FT from 20k ==="
echo "  BASE_CKPT:    $BASE_CKPT"
echo "  DATASET_PATH: $DATASET_PATH"
echo "  data_config:  libero"
echo "  tag:          libero"
echo "  steps:        $MAX_STEPS (save every $SAVE_STEPS)"
echo "  batch:        $BATCH_SIZE x accum $GRAD_ACCUM x gpus $NUM_GPUS"
echo "  split:        $DATA_SPLIT"
echo "  video:        $VIDEO_BACKEND"
echo "  tune_llm:     $TUNE_LLM"
echo "  unified tag:  ${UNIFIED_TAG:-<none>}"
echo "  output:       $OUT"
echo "  log:          $LOG"

TUNE_ARGS=(--tune-llm)
if [[ "$TUNE_LLM" == "0" ]]; then
  TUNE_ARGS=(--no-tune-llm)
fi

EXTRA_ARGS=()
if [[ -n "$UNIFIED_TAG" ]]; then
  EXTRA_ARGS+=(--unified_embodiment_tag "$UNIFIED_TAG")
fi

"$PYTHON" -u scripts/dual_system_train.py \
  --bridge_type end2end \
  --dataset-path "$DATASET_PATH" \
  --data-config libero \
  --embodiment_tag libero \
  --data_split "$DATA_SPLIT" \
  --num-gpus "$NUM_GPUS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --dataloader-num-workers "$WORKERS" \
  --video_backend "$VIDEO_BACKEND" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --base_model_path "$BASE_CKPT" \
  --no-compute-bridge-loss \
  "${TUNE_ARGS[@]}" \
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
  "${EXTRA_ARGS[@]}" \
  --report_to tensorboard \
  --output_dir "$OUT" 2>&1 | tee "$LOG"

echo "=== smoke done: $OUT ==="
if [[ -f "$OUT/experiment_cfg/metadata.json" ]]; then
  echo "metadata:"
  "$PYTHON" - "$OUT/experiment_cfg/metadata.json" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
print("  tags:", sorted(data))
for tag, md in data.items():
    print(f"  {tag}: state={sorted(md['statistics']['state'])}")
    print(f"  {tag}: action={sorted(md['statistics']['action'])}")
PY
else
  echo "WARNING: no experiment_cfg/metadata.json under $OUT" >&2
fi
