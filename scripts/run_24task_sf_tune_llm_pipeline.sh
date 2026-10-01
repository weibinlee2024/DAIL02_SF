#!/usr/bin/env bash
# 24-task pipeline: golden pretrain 20k (save 5k) → end2end finetune test 10k (save 10k)
# --tune-llm + state-future. AMP, no MODEL_BF16.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP=${EXP:-"/media/weibin/weibin02/DIAL/20260918"}
PIPELINE_LOG=${PIPELINE_LOG:-"$EXP/pipeline_24task_sf_tune_llm.log"}
mkdir -p "$EXP" "$(dirname "$PIPELINE_LOG")"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPELINE_LOG"; }

export DATA_ROOT=${DATA_ROOT:-"/media/weibin/weibin01/DIAL_DATASETS/LeRobot-AugPosRot-Correct"}
export PYTHON=${PYTHON:-/home/weibin/anaconda3/envs/dial/bin/python}
export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-2}
export GRAD_ACCUM=${GRAD_ACCUM:-32}

OUT_PRE=${OUT_PRE:-"$EXP/pretrain-golden-sf-24task-20k-tune-llm"}
OUT_FT=${OUT_FT:-"$EXP/finetune-end2end-sf-24task-tune-llm"}
LOG_PRE=${LOG_PRE:-"$EXP/pretrain_24task_sf_tune_llm.log"}
LOG_FT=${LOG_FT:-"$EXP/finetune_24task_sf_tune_llm.log"}

log "=== 24TASK PIPELINE START pre=20k/save5k ft=10k/save10k tune-llm SF AMP ==="
log "OUT_PRE=$OUT_PRE"
log "OUT_FT=$OUT_FT"

log ">>> PRETRAIN 20k save every 5k"
OUT="$OUT_PRE" LOG="$LOG_PRE" MAX_STEPS=20000 SAVE_STEPS=5000 \
  bash "$ROOT/scripts/run_24task_pretrain_golden_sf_tune_llm_20k.sh"
rc_pre=$?
log "pretrain rc=$rc_pre"
if [[ $rc_pre -ne 0 ]]; then
  log "PIPELINE ABORT: pretrain failed"
  exit $rc_pre
fi

CKPT="$OUT_PRE/checkpoint-20000"
if [[ ! -d "$CKPT" ]]; then
  CKPT=$(ls -d "$OUT_PRE"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
fi
if [[ -z "${CKPT:-}" || ! -d "$CKPT" ]]; then
  log "PIPELINE ABORT: no pretrain checkpoint under $OUT_PRE"
  exit 2
fi

log ">>> FINETUNE TEST 10k from $CKPT"
BASE_CKPT="$CKPT" OUT="$OUT_FT" LOG="$LOG_FT" MAX_STEPS=10000 SAVE_STEPS=10000 \
  bash "$ROOT/scripts/run_24task_finetune_end2end_sf_tune_llm_test.sh"
rc_ft=$?
log "finetune rc=$rc_ft"
log "=== PIPELINE DONE rc_pre=$rc_pre rc_ft=$rc_ft ==="
exit $rc_ft
