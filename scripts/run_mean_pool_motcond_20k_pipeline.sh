#!/usr/bin/env bash
# Dense-CA MoT cond experts + mean_pool SF decoder:
#   golden 20k pretrain → end2end 10k finetune (+ offline eval).
# Same protocol as run_dense_mot_cond_kv_20k_pipeline.sh except decoder=mean_pool.
#
# Usage:
#   bash scripts/run_mean_pool_motcond_20k_pipeline.sh
#   STAGE=pre|ft|all bash scripts/run_mean_pool_motcond_20k_pipeline.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-"/media/weibin/weibin02/LeRobot-AugPosRot-Correct"}
PYTHON=${PYTHON:-/home/weibin/anaconda3/envs/dial/bin/python}
W=${SF_DUAL_W:-0.05}
WTAG=$(echo "$W" | tr '.' 'p')
OUT_ROOT=${ALIGN_OUT_ROOT:-"/media/weibin/weibin02/DIAL_MoT"}
STAGE=${STAGE:-all}

if [[ -n "${SMOKE_STEPS:-}" ]]; then
  PRE_STEPS=${SMOKE_STEPS}
  PRE_SAVE=${SMOKE_STEPS}
else
  PRE_STEPS=${PRE_STEPS:-20000}
  PRE_SAVE=${PRE_SAVE:-5000}
fi
FT_STEPS=${FT_STEPS:-10000}
FT_SAVE=${FT_SAVE:-5000}

BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-16}
NUM_GPUS=${NUM_GPUS:-1}
TRAJS=${TRAJS:-10}
BASE_CFG=${BASE_CFG:-"$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json"}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop}

PRE_OUT="${OUT_ROOT}/mean_pool_motcond_w${WTAG}_tuneLLM_L16_pre_20k"
FT_OUT="${OUT_ROOT}/mean_pool_motcond_w${WTAG}_tuneLLM_L16_ft_10k"
PRE_CKPT="${PRE_OUT}/checkpoint-${PRE_STEPS}"
PRE_LOG="${PRE_OUT}.log"
FT_LOG="${FT_OUT}.log"

export MODEL_BF16=${MODEL_BF16:-0}
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/media/weibin/weibin02/hf_home_dial}
export TMPDIR=${TMPDIR:-/media/weibin/weibin02/tmp_dial}
mkdir -p "$HF_HOME" "$TMPDIR"
[[ -e "$HF_HOME/hub" ]] || ln -sf /home/weibin/.cache/huggingface/hub "$HF_HOME/hub" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true

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

run_eval() {
  local ckpt="$1" log="$2" label="$3"
  if [[ -n "${SMOKE_STEPS:-}" && "$PRE_STEPS" -lt 100 ]]; then
    echo "SMOKE skip eval ($label)" | tee -a "$log"
    return 0
  fi
  local eval_dir="$ckpt/eval_action_goal_loss"
  mkdir -p "$eval_dir"
  echo "=== offline eval $label $(date -Is) ckpt=$ckpt ===" | tee -a "$log"
  PYTHONBREAKPOINT=0 "$PYTHON" -u scripts/eval_policy_dial.py \
    --dataset-path "${DS[@]}" \
    --model_path "$ckpt" \
    --data-config "$DATA_CONFIG" \
    --embodiment_tag gr1 \
    --trajs "$TRAJS" \
    --save_results_path "$eval_dir" \
    --plot_state \
    --vis_pca >> "$log" 2>&1
  local erc=$?
  echo "EVAL $label rc=$erc $(date -Is)" | tee -a "$log"
  if [[ -f "$eval_dir/results.json" ]]; then
    echo "=== results.json ($label) ===" | tee -a "$log"
    cat "$eval_dir/results.json" | tee -a "$log"
  fi
  return $erc
}

run_pre() {
  if [[ -d "$PRE_CKPT" ]]; then
    echo "SKIP pre: $PRE_CKPT exists"
    return 0
  fi
  mkdir -p "$PRE_OUT" "$(dirname "$PRE_LOG")"
  echo "=== mean_pool+motcond PRE golden steps=$PRE_STEPS w=$W start $(date -Is) OUT=$PRE_OUT ===" | tee -a "$PRE_LOG"
  "$PYTHON" -u scripts/dual_system_train.py \
    --bridge_type golden \
    --dataset-path "${DS[@]}" \
    --data-config "$DATA_CONFIG" \
    --embodiment_tag gr1 \
    --data_split "[:-10]" \
    --num-gpus "$NUM_GPUS" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --dataloader-num-workers 2 --max-steps "$PRE_STEPS" --save-steps "$PRE_SAVE" \
    --base_model_path "$BASE_CFG" \
    --compute-bridge-loss --tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
    --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
    --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
    --report_to tensorboard \
    --use-state-future --compute-state-future-loss --state-future-loss-weight "$W" \
    --state-future-decoder mean_pool --state-future-dim 30 \
    --loss-combine-mode additive \
    --feed-state-future-to-dit \
    --sf-dit-repr shat \
    --use-mot-cond-experts --mot-include-sf --mot-sf-num-tokens 17 \
    --output_dir "$PRE_OUT" >> "$PRE_LOG" 2>&1
  local rc=$?
  echo "TRAIN pre rc=$rc $(date -Is)" | tee -a "$PRE_LOG"
  [[ $rc -eq 0 ]] || return $rc
  run_eval "$PRE_CKPT" "$PRE_LOG" "pre"
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
  mkdir -p "$FT_OUT" "$(dirname "$FT_LOG")"
  echo "=== mean_pool+motcond FT end2end steps=$FT_STEPS base=$base_ckpt start $(date -Is) OUT=$FT_OUT ===" | tee -a "$FT_LOG"
  "$PYTHON" -u scripts/dual_system_train.py \
    --bridge_type end2end \
    --dataset-path "${DS[@]}" \
    --data-config "$DATA_CONFIG" \
    --embodiment_tag gr1 \
    --data_split "[:-10]" \
    --num-gpus "$NUM_GPUS" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --dataloader-num-workers 2 --max-steps "$FT_STEPS" --save-steps "$FT_SAVE" \
    --base_model_path "$base_ckpt" \
    --compute-bridge-loss --tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
    --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
    --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
    --report_to tensorboard \
    --use-state-future --compute-state-future-loss --state-future-loss-weight "$W" \
    --state-future-decoder mean_pool --state-future-dim 30 \
    --loss-combine-mode additive \
    --feed-state-future-to-dit \
    --sf-dit-repr shat \
    --use-mot-cond-experts --mot-include-sf --mot-sf-num-tokens 17 \
    --output_dir "$FT_OUT" >> "$FT_LOG" 2>&1
  local rc=$?
  echo "TRAIN ft rc=$rc $(date -Is)" | tee -a "$FT_LOG"
  [[ $rc -eq 0 ]] || return $rc
  run_eval "$FT_OUT/checkpoint-${FT_STEPS}" "$FT_LOG" "ft"
}

cd "$ROOT"
mkdir -p "$OUT_ROOT"
case "$STAGE" in
  pre) run_pre ;;
  ft)  run_ft ;;
  all)
    run_pre || exit $?
    run_ft  || exit $?
    ;;
  *) echo "STAGE=$STAGE invalid (pre|ft|all)"; exit 1 ;;
esac

echo "=== mean_pool+motcond pipeline done $(date -Is) ==="
# write compact pointer for later online SR
python3 - <<PY
import json
from pathlib import Path
ft = Path("$FT_OUT") / "checkpoint-${FT_STEPS}"
pre = Path("$PRE_CKPT")
card = {
  "decoder": "mean_pool",
  "use_mot_cond_experts": True,
  "sf_dit_repr": "shat",
  "w": float("$W"),
  "pre_ckpt": str(pre) if pre.is_dir() else None,
  "ft_ckpt": str(ft) if ft.is_dir() else None,
  "compare_to_query_ca_motcond": "/media/weibin/weibin02/DIAL_MoT/dense_motcond_w0p05_tuneLLM_L16_ft_10k/checkpoint-10000",
}
Path("$OUT_ROOT/mean_pool_motcond_w${WTAG}_card.json").write_text(json.dumps(card, indent=2))
print(json.dumps(card, indent=2))
PY
