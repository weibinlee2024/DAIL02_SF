#!/usr/bin/env bash
# SF weight debug: 2k-step sweep over w ∈ {0, 0.05, 0.1} with additive loss
# L = action + bridge + w·sf  (loss_combine_mode=additive).
# Same 6 tasks / query_ca / tune-llm as the 10k run. Does NOT start 24-task.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-"/media/weibin/Data/DIAL_DATASETS/LeRobot-AugPosRot-Correct"}
PYTHON=${PYTHON:-/home/weibin/anaconda3/envs/dial/bin/python}
BASE_OUT=${SF_WSWEEP_OUT:-"/media/weibin/weibin02/DIAL/20260922/sf_weight_debug/w_sweep_2k"}
LOG_DIR=${SF_WSWEEP_LOGDIR:-"/media/weibin/weibin02/DIAL/20260922/sf_weight_debug"}
MAX_STEPS=${SF_WSWEEP_STEPS:-2000}
SAVE_STEPS=${SF_WSWEEP_SAVE:-2000}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-32}
NUM_GPUS=${NUM_GPUS:-1}
TRAJS=${TRAJS:-10}
BASE_CFG=${BASE_CFG:-"$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json"}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop}
WEIGHTS=(${SF_WSWEEP_WEIGHTS:-0 0.05 0.1})
mkdir -p "$BASE_OUT" "$LOG_DIR"

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
    echo "ERROR: missing dataset $p"
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

SUMMARY="$LOG_DIR/w_sweep_2k_summary.jsonl"
: > "$SUMMARY"
MASTER_LOG="$LOG_DIR/w_sweep_2k_master.log"
echo "=== SF w-sweep 2k start $(date -Is) weights=${WEIGHTS[*]} mode=additive ===" | tee -a "$MASTER_LOG"

for W in "${WEIGHTS[@]}"; do
  # sanitize folder name: 0.05 -> 0p05
  WTAG=$(echo "$W" | tr '.' 'p')
  OUT="$BASE_OUT/w${WTAG}"
  LOG="$LOG_DIR/w_sweep_2k_w${WTAG}.log"
  mkdir -p "$OUT"
  echo "----- w=$W out=$OUT $(date -Is) -----" | tee -a "$MASTER_LOG" "$LOG"

  # Clear leaked OUT/LOG env from other shells
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
    --output_dir "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "TRAIN w=$W rc=$rc $(date -Is)" | tee -a "$MASTER_LOG" "$LOG"
  if [[ $rc -ne 0 ]]; then
    echo "{\"w\": $W, \"train_rc\": $rc, \"status\": \"train_failed\"}" >> "$SUMMARY"
    continue
  fi

  CKPT="$OUT/checkpoint-${MAX_STEPS}"
  if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: missing $CKPT" | tee -a "$MASTER_LOG" "$LOG"
    echo "{\"w\": $W, \"train_rc\": 0, \"status\": \"missing_ckpt\"}" >> "$SUMMARY"
    continue
  fi

  EVAL_DIR="$CKPT/eval_action_goal_loss"
  mkdir -p "$EVAL_DIR"
  echo "=== offline eval w=$W $(date -Is) ===" | tee -a "$MASTER_LOG" "$LOG"
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
  echo "EVAL w=$W rc=$erc $(date -Is)" | tee -a "$MASTER_LOG" "$LOG"

  # Pull key metrics into summary line
  RES="$EVAL_DIR/results.json"
  if [[ -f "$RES" ]]; then
    "$PYTHON" - <<PY >> "$SUMMARY"
import json
w=$W
with open("$RES") as f:
    r=json.load(f)
# results.json shape varies; keep whole payload under "eval"
print(json.dumps({"w": w, "train_rc": 0, "eval_rc": $erc, "status": "ok", "ckpt": "$CKPT", "eval": r}, default=str))
PY
  else
    echo "{\"w\": $W, \"train_rc\": 0, \"eval_rc\": $erc, \"status\": \"no_results_json\"}" >> "$SUMMARY"
  fi
done

echo "=== SF w-sweep 2k done $(date -Is) summary=$SUMMARY ===" | tee -a "$MASTER_LOG"
cat "$SUMMARY" | tee -a "$MASTER_LOG"
