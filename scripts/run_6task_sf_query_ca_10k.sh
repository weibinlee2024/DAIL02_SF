#!/usr/bin/env bash
# 6-task golden 10k: query_ca SF + tune-LLM + select_layer 12, then offline eval_loss.
# Same task split as previous quick6task10k (3 hard + 3 easy).
# Reference (old tune mean-pool @10k, 6-task mean):
#   action_mse 0.1381  (no-tune 0.1481, −6.8%)
#   bridge_loss 0.6695 (no-tune 0.9129, −26.7%)
# Train from scratch (do NOT resume the 2k query_ca ckpt; cosine is scheduled on max_steps).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-"/media/weibin/Data/DIAL_DATASETS/LeRobot-AugPosRot-Correct"}
PYTHON=${PYTHON:-/home/weibin/anaconda3/envs/dial/bin/python}
# Do not reuse generic OUT/LOG/SAVE_STEPS — those leak from older eval/finetune shells.
OUT=${SF6_OUT:-"/media/weibin/weibin02/DIAL/20260922/pretrain-golden-sf-6task-10k-query-ca-tune-llm"}
LOG=${SF6_LOG:-"/media/weibin/weibin02/DIAL/20260922/pretrain_6task_sf_query_ca_10k.log"}
MAX_STEPS=${SF6_MAX_STEPS:-10000}
SAVE_STEPS=${SF6_SAVE_STEPS:-5000}
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
  if [[ ! -d "$p" ]]; then
    echo "ERROR: missing dataset $p" | tee -a "$LOG"
    exit 1
  fi
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

echo "=== 6task SF query_ca 10k (tune-llm, AMP bf16) start $(date -Is) ===" | tee -a "$LOG"
echo "DATA_ROOT=$DATA_ROOT OUT=$OUT MAX_STEPS=$MAX_STEPS decoder=query_ca dim=30 select_layer=12" | tee -a "$LOG"
echo "REF old tune mean-pool@10k: action_mse=0.1381 bridge_loss=0.6695 (6-task mean)" | tee -a "$LOG"

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
  --use-state-future --compute-state-future-loss --state-future-loss-weight 0.5 \
  --state-future-decoder query_ca --state-future-dim 30 \
  --output_dir "$OUT" >> "$LOG" 2>&1
rc=$?
echo "6TASK-PRETRAIN-QUERY-CA DONE rc=$rc $(date -Is)" | tee -a "$LOG"
if [[ $rc -ne 0 ]]; then
  exit $rc
fi

CKPT="$OUT/checkpoint-${MAX_STEPS}"
if [[ ! -d "$CKPT" ]]; then
  echo "ERROR: missing $CKPT" | tee -a "$LOG"
  exit 1
fi

EVAL_DIR="$CKPT/eval_action_goal_loss"
mkdir -p "$EVAL_DIR"
echo "=== offline eval_loss start $(date -Is) ckpt=$CKPT trajs=$TRAJS ===" | tee -a "$LOG"
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
echo "6TASK-OFFLINE-EVAL DONE rc=$erc $(date -Is)" | tee -a "$LOG"
if [[ -f "$EVAL_DIR/results.json" ]]; then
  echo "=== results.json ===" | tee -a "$LOG"
  cat "$EVAL_DIR/results.json" | tee -a "$LOG"
fi
exit $erc
