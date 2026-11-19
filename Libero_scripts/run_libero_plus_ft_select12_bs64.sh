#!/usr/bin/env bash
# LIBERO-plus FT from the RoboCasa 20k init with the requested consistency config:
#   select_layer=12, effective batch = 64 (2 x 32), tune_llm, UNIFIED_TAG=gr1.
#
# Note: the 20k checkpoint config uses select_layer=36. select_layer=12 keeps
# only the first 12 LLM layers and discards layers 12-35; this is an explicit
# tradeoff for memory/speed and should be compared with select_layer=36 later.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export BASE_CKPT="${BASE_CKPT:-/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/DIAL02/outputs/20261102/checkpoint_copies/policy_meal_sf36L_e2e_ckpt20000_meanpool_infer}"
export OUT="${OUT:-/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/DIAL02/outputs/20261112/libero_plus_ft_select12_bs64_20k}"
export SELECT_LAYER="${SELECT_LAYER:-12}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-32}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export SAVE_STEPS="${SAVE_STEPS:-2500}"
export WORKERS="${WORKERS:-4}"
export MODEL_BF16="${MODEL_BF16:-1}"
export UNIFIED_TAG="${UNIFIED_TAG:-gr1}"
export DATA_SPLIT="${DATA_SPLIT:-[:]}"
export TASK_INDICES="${TASK_INDICES:-0 1 2 3 4 5 6 7}"
exec bash "$ROOT/scripts/run_libero_plus_ft_20k_unified.sh"
