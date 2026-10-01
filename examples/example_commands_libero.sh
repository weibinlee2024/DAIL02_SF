#!/usr/bin/env bash
# Example LIBERO training commands (horizon=8, data-config=libero).
# Adjust LIBERO_DIR / paths to your converted LeRobot roots.
#
# Usage (from repo root):
#   source examples/example_commands_libero.sh   # defines helpers, does not train
#   libero_smoke_decoupled
#   libero_pretrain_decoupled
#   libero_finetune_end2end <stage1_ckpt>

LIBERO_DIR="${LIBERO_DIR:-data/libero}"
OUT_ROOT="${OUT_ROOT:-outputs}"

COMMON_FLAGS=(
  --compute-bridge-loss
  --tune-llm
  --no-tune-visual
  --tune-projector
  --tune-diffusion-model
  --no-tune-bridge-visual
  --no-tune-bridge-goal
  --tune-bridge-embedding
  --select_layer 36
  --use_image_type_embedding
  --ignore_lang_prefix
  --data-config libero
  --embodiment-tag libero
)

libero_datasets_one() {
  echo "${LIBERO_DIR}/libero_spatial"
}

libero_datasets_all() {
  echo \
    "${LIBERO_DIR}/libero_spatial" \
    "${LIBERO_DIR}/libero_object" \
    "${LIBERO_DIR}/libero_goal" \
    "${LIBERO_DIR}/libero_10"
}

libero_smoke_decoupled() {
  bash examples/train.sh decoupled \
    --dataset-path "$(libero_datasets_one)" \
    --output-dir "${OUT_ROOT}/libero-smoke-decoupled" \
    --batch-size 4 \
    --max-steps 100 \
    --save-steps 100 \
    --num-gpus 1 \
    "${COMMON_FLAGS[@]}" \
    "$@"
}

libero_pretrain_decoupled() {
  # shellcheck disable=SC2046
  bash examples/train.sh decoupled \
    --dataset-path $(libero_datasets_all) \
    --data-config libero libero libero libero \
    --embodiment-tag libero libero libero libero \
    --output-dir "${OUT_ROOT}/libero-pretrain-decoupled" \
    "${COMMON_FLAGS[@]}" \
    "$@"
}

libero_finetune_end2end() {
  local ckpt=${1:?"Usage: libero_finetune_end2end <stage1_checkpoint_dir>"}
  shift || true
  # shellcheck disable=SC2046
  bash examples/train.sh end2end \
    --base-model-path "$ckpt" \
    --dataset-path $(libero_datasets_all) \
    --data-config libero libero libero libero \
    --embodiment-tag libero libero libero libero \
    --output-dir "${OUT_ROOT}/libero-finetune-end2end" \
    --use-separate-projector-for-loss \
    "${COMMON_FLAGS[@]}" \
    "$@"
}

echo "Loaded LIBERO helpers: libero_smoke_decoupled / libero_pretrain_decoupled / libero_finetune_end2end"
echo "LIBERO_DIR=$LIBERO_DIR"
