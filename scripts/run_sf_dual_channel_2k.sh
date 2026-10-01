#!/usr/bin/env bash
# Dual-channel foresight → DiT: 6task 2k with feed_state_future_to_dit.
# Usage:
#   SF_DUAL_W=1 bash scripts/run_sf_dual_channel_2k.sh
#   SF_DUAL_W=0.05 bash scripts/run_sf_dual_channel_2k.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-"/media/weibin/Data/DIAL_DATASETS/LeRobot-AugPosRot-Correct"}
PYTHON=${PYTHON:-/home/weibin/anaconda3/envs/dial/bin/python}
W=${SF_DUAL_W:-0.05}
WTAG=$(echo "$W" | tr '.' 'p')
REPR=${SF_DIT_REPR:-shat}
OUT=${SF_DUAL_OUT:-"/media/weibin/weibin02/DIAL/20260922/sf_weight_debug/dual_channel_2k/${REPR}_w${WTAG}"}
LOG=${SF_DUAL_LOG:-"/media/weibin/weibin02/DIAL/20260922/sf_weight_debug/dual_channel_2k/${REPR}_w${WTAG}.log"}
MAX_STEPS=${SF_DUAL_STEPS:-2000}
SAVE_STEPS=${SF_DUAL_SAVE:-2000}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-32}
NUM_GPUS=${NUM_GPUS:-1}
TRAJS=${TRAJS:-10}
BASE_CFG=${BASE_CFG:-"$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json"}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop}
mkdir -p "$OUT" "$(dirname "$LOG")"

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

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/home/weibin/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset MODEL_BF16
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== dual-channel 2k w=$W repr=$REPR feed_to_dit=True start $(date -Is) OUT=$OUT ===" | tee -a "$LOG"

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
  --sf-dit-repr "$REPR" \
  --output_dir "$OUT" >> "$LOG" 2>&1
rc=$?
echo "TRAIN dual w=$W repr=$REPR rc=$rc $(date -Is)" | tee -a "$LOG"
[[ $rc -eq 0 ]] || exit $rc

CKPT="$OUT/checkpoint-${MAX_STEPS}"
EVAL_DIR="$CKPT/eval_action_goal_loss"
mkdir -p "$EVAL_DIR"
echo "=== offline eval dual w=$W repr=$REPR $(date -Is) ===" | tee -a "$LOG"
PYTHONBREAKPOINT=0 "$PYTHON" -u scripts/eval_policy_dial.py \
  --dataset-path "${DS[@]}" \
  --model_path "$CKPT" \
  --data-config "$DATA_CONFIG" \
  --embodiment_tag gr1 \
  --trajs "$TRAJS" \
  --save_results_path "$EVAL_DIR" \
  --plot_state \
  --vis_pca >> "$LOG" 2>&1
erc=$?
echo "EVAL dual w=$W repr=$REPR rc=$erc $(date -Is)" | tee -a "$LOG"
if [[ -f "$EVAL_DIR/results.json" ]]; then
  echo "=== results.json ===" | tee -a "$LOG"
  cat "$EVAL_DIR/results.json" | tee -a "$LOG"
fi
exit $erc
