#!/usr/bin/env bash
# Aligned 2k pair: dense DiT+Ŝ vs MoT System-1, same protocol as prior dense+tune-llm.
# No architectural derating (16 layers, tune-llm). Shared mem knobs only.
#
# Usage:
#   bash scripts/run_mot_vs_dense_aligned_2k.sh              # both: dense then MoT
#   RUN_WHICH=dense bash scripts/run_mot_vs_dense_aligned_2k.sh
#   RUN_WHICH=mot   bash scripts/run_mot_vs_dense_aligned_2k.sh
#   SMOKE_STEPS=5   bash ...   # short OOM check
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-"/media/weibin/Data/DIAL_DATASETS/LeRobot-AugPosRot-Correct"}
PYTHON=${PYTHON:-/home/weibin/anaconda3/envs/dial/bin/python}
W=${SF_DUAL_W:-0.05}
WTAG=$(echo "$W" | tr '.' 'p')
OUT_ROOT=${ALIGN_OUT_ROOT:-"/media/weibin/C4E60209E601FC84/DIAL/20260928/sf_weight_debug/mot_vs_dense_aligned_2k"}
MAX_STEPS=${MOT_STEPS:-${SMOKE_STEPS:-2000}}
SAVE_STEPS=${MOT_SAVE:-$MAX_STEPS}
# Match prior dense effective batch (2×32); microbatch 1 for MoT VRAM headroom
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-64}
NUM_GPUS=${NUM_GPUS:-1}
TRAJS=${TRAJS:-10}
BASE_CFG=${BASE_CFG:-"$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json"}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop}
RUN_WHICH=${RUN_WHICH:-both}   # dense | mot | both
# Full MoT depth (no L8 derate). MODEL_BF16 is a mem knob shared by both arms.
export MODEL_BF16=${MODEL_BF16:-1}
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/home/weibin/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASKS=(
  PosttrainPnPNovelFromPlacematToPlateSplitA
  PosttrainPnPNovelFromCuttingboardToPotSplitA
  PosttrainPnPNovelFromCuttingboardToPanSplitA
  PosttrainPnPNovelFromPlateToBowlSplitA
  PosttrainPnPNovelFromTrayToCardboardboxSplitA
  PosttrainPnPNovelFromTrayToPlateSplitA
)
DS=()
for t in "${TASKS[@]}"; do
  p="$DATA_ROOT/gr1_unified.$t"
  [[ -d "$p" ]] || { echo "ERROR: missing $p"; exit 1; }
  DS+=("$p")
done

run_one() {
  local mode="$1"   # dense | mot
  local OUT LOG EXTRA
  if [[ "$mode" == "dense" ]]; then
    OUT="${OUT_ROOT}/dense_shat_w${WTAG}_tuneLLM_L16"
    LOG="${OUT}.log"
    EXTRA=( )
  else
    OUT="${OUT_ROOT}/mot_shat_w${WTAG}_tuneLLM_L16"
    LOG="${OUT}.log"
    EXTRA=( --use-mot-system1 --mot-include-sf --mot-sf-num-tokens 17 )
    # explicit 16 = config default; do NOT pass 8
    EXTRA+=( --mot-num-layers 16 )
  fi
  mkdir -p "$OUT" "$(dirname "$LOG")"
  echo "=== ALIGNED $mode 2k w=$W bs=$BATCH_SIZE accum=$GRAD_ACCUM MODEL_BF16=$MODEL_BF16 tune-llm L16 start $(date -Is) OUT=$OUT ===" | tee -a "$LOG"

  env -u OUT -u LOG -u SAVE_STEPS -u MAX_STEPS \
  "$PYTHON" -u scripts/dual_system_train.py \
    --bridge_type golden \
    --dataset-path "${DS[@]}" \
    --data-config "$DATA_CONFIG" \
    --embodiment_tag gr1 \
    --data_split "[:-10]" \
    --num-gpus "$NUM_GPUS" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --dataloader-num-workers 2 --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" \
    --base_model_path "$BASE_CFG" \
    --compute-bridge-loss --tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
    --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
    --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
    --report_to tensorboard \
    --use-state-future --compute-state-future-loss --state-future-loss-weight "$W" \
    --state-future-decoder query_ca --state-future-dim 30 \
    --loss-combine-mode additive \
    --feed-state-future-to-dit \
    --sf-dit-repr shat \
    "${EXTRA[@]}" \
    --output_dir "$OUT" >> "$LOG" 2>&1
  local rc=$?
  echo "TRAIN $mode w=$W rc=$rc $(date -Is)" | tee -a "$LOG"
  [[ $rc -eq 0 ]] || return $rc

  # skip offline eval on smoke
  if [[ "${SMOKE_STEPS:-}" != "" && "$MAX_STEPS" -lt 100 ]]; then
    echo "SMOKE skip eval steps=$MAX_STEPS" | tee -a "$LOG"
    return 0
  fi

  local CKPT="$OUT/checkpoint-${MAX_STEPS}"
  local EVAL_DIR="$CKPT/eval_action_goal_loss"
  mkdir -p "$EVAL_DIR"
  echo "=== offline eval $mode w=$W $(date -Is) ===" | tee -a "$LOG"
  PYTHONBREAKPOINT=0 "$PYTHON" -u scripts/eval_policy_dial.py \
    --dataset-path "${DS[@]}" \
    --model_path "$CKPT" \
    --data-config "$DATA_CONFIG" \
    --embodiment_tag gr1 \
    --trajs "$TRAJS" \
    --save_results_path "$EVAL_DIR" \
    --plot_state \
    --vis_pca >> "$LOG" 2>&1
  local erc=$?
  echo "EVAL $mode w=$W rc=$erc $(date -Is)" | tee -a "$LOG"
  if [[ -f "$EVAL_DIR/results.json" ]]; then
    echo "=== results.json ($mode) ===" | tee -a "$LOG"
    cat "$EVAL_DIR/results.json" | tee -a "$LOG"
  fi
  return $erc
}

cd "$ROOT"
mkdir -p "$OUT_ROOT"
case "$RUN_WHICH" in
  dense) run_one dense ;;
  mot)   run_one mot ;;
  both)
    run_one dense || exit $?
    run_one mot   || exit $?
    ;;
  *) echo "RUN_WHICH=$RUN_WHICH invalid (dense|mot|both)"; exit 1 ;;
esac
