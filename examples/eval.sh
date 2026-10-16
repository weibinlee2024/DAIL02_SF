#!/bin/bash
# DIAL Evaluation Script
# Usage: bash examples/eval.sh <model_path> <eval_type>
#
# EVAL_TYPE:
#   id              - In-distribution (24 tasks: 6 PnPClose + 18 PosttrainPnPNovel SplitA)
#   ood_object_appearance  - OOD unseen object appearances (18 EvalPnPNovel SplitB)
#   ood_container_combination   - OOD unseen source-target containers (14 PretrainPnPNovel SplitA)
#   ood_object_type      - OOD unseen object types (32 PretrainPnPBase SplitA)
#
# Environment variables:
#   PORT         - Server port (default: 50051)
#   N_EPISODES   - Episodes per task (default: 50)
#   DATA_CONFIG  - (default: fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop)
#   SF_EVAL_MODE - Online SF ablation: correct | zero | swap (default: correct)
#   SF_SWAP_BANK_MAX - swap token bank size (default: 64)
#   EVAL_SUBDIR  - Override results subdirectory (optional)

set -euo pipefail

export PATH=/home/weibin/anaconda3/envs/dial/bin:$PATH

MODEL_PATH=${1:?"Usage: bash examples/eval.sh <model_path> <eval_type>"}
EVAL_TYPE=${2:?"Eval type: id | id6 | ood_object_appearance | ood_container_combination | ood_object_type"}
PORT=${PORT:-50051}
N_EPISODES=${N_EPISODES:-50}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop}
SF_EVAL_MODE=${SF_EVAL_MODE:-correct}
SF_SWAP_BANK_MAX=${SF_SWAP_BANK_MAX:-64}
case "$SF_EVAL_MODE" in
  correct|zero|swap) ;;
  *) echo "Error: SF_EVAL_MODE must be correct|zero|swap, got '$SF_EVAL_MODE'"; exit 1 ;;
esac

case "$EVAL_TYPE" in
  id)
    _DEFAULT_EVAL_SUBDIR="evaluation_sim_id_${N_EPISODES}eps"
    task_names=(
        gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
    )
    EVAL_SUBDIR="${EVAL_SUBDIR:-$_DEFAULT_EVAL_SUBDIR}"
    ;;
  id6)
    # 6 dual-channel offline tasks (PosttrainPnPNovel SplitA subset)
    _DEFAULT_EVAL_SUBDIR="evaluation_sim_id6_${N_EPISODES}eps"
    task_names=(
        gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
    )
    EVAL_SUBDIR="${EVAL_SUBDIR:-$_DEFAULT_EVAL_SUBDIR}"
    ;;
  ood_object_appearance)
    _DEFAULT_EVAL_SUBDIR="evaluation_sim_ood_object_appearance_${N_EPISODES}eps"
    EVAL_SUBDIR="${EVAL_SUBDIR:-$_DEFAULT_EVAL_SUBDIR}"
    task_names=(
        gr1_unified/EvalPnPNovelFromCuttingboardToBasketSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromCuttingboardToCardboardboxSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromCuttingboardToPanSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromCuttingboardToPotSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromCuttingboardToTieredbasketSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlacematToBasketSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlacematToBowlSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlacematToPlateSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlacematToTieredshelfSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlateToBowlSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlateToCardboardboxSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlateToPanSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlateToPlateSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToCardboardboxSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToPlateSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToPotSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToTieredbasketSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToTieredshelfSplitB_GR1ArmsAndWaistFourierHands_Env
    ) ;;
  ood_container_combination)
    EVAL_SUBDIR="evaluation_sim_ood_container_combination_${N_EPISODES}eps"
    task_names=(
        gr1_unified/PretrainPnPNovelFromCuttingboardToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromCuttingboardToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromCuttingboardToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromTrayToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromTrayToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromTrayToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlateToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlateToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlateToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlateToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlacematToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlacematToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlacematToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlacematToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
    ) ;;
  ood_object_type)
    EVAL_SUBDIR="evaluation_sim_ood_object_type_${N_EPISODES}eps"
    task_names=(
        gr1_unified/PretrainPnPBaseFromCuttingboardToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
    ) ;;
  *) echo "Error: Unknown EVAL_TYPE '$EVAL_TYPE'."; exit 1 ;;
esac

# Append sf mode to subdir unless caller overrode EVAL_SUBDIR explicitly
if [[ -z "${EVAL_SUBDIR_SET:-}" && "$SF_EVAL_MODE" != "correct" ]]; then
  EVAL_SUBDIR="${EVAL_SUBDIR}_sf_${SF_EVAL_MODE}"
elif [[ -z "${EVAL_SUBDIR_SET:-}" && "$SF_EVAL_MODE" == "correct" && "${FORCE_SF_TAG:-0}" == "1" ]]; then
  EVAL_SUBDIR="${EVAL_SUBDIR}_sf_correct"
fi

echo "=============================================="
echo "DIAL Evaluation"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  EVAL_TYPE:  $EVAL_TYPE ($EVAL_SUBDIR)"
echo "  PORT:       $PORT"
echo "  N_EPISODES: $N_EPISODES"
echo "  SF_EVAL_MODE: $SF_EVAL_MODE"
echo "  TASKS:      ${#task_names[@]}"
echo "=============================================="

mkdir -p "$MODEL_PATH/$EVAL_SUBDIR"
CLIENT_LOG="$MODEL_PATH/$EVAL_SUBDIR/client.log"

nohup python3 -u scripts/inference_service_dial.py --server \
    --embodiment_tag gr1 \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --data_config "$DATA_CONFIG" \
    --sf_eval_mode "$SF_EVAL_MODE" \
    --sf_swap_bank_max "$SF_SWAP_BANK_MAX" \
    > "$MODEL_PATH/$EVAL_SUBDIR/server.log" 2>&1 &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID), waiting until ready (max ${SERVER_WAIT_SEC:-900}s)..."
ready=0
waited=0
SERVER_WAIT_SEC=${SERVER_WAIT_SEC:-900}
while [[ $waited -lt $SERVER_WAIT_SEC ]]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server PID $SERVER_PID exited before ready; see $MODEL_PATH/$EVAL_SUBDIR/server.log"
    break
  fi
  if rg -q 'Server is ready and listening' "$MODEL_PATH/$EVAL_SUBDIR/server.log" 2>/dev/null; then
    ready=1
    echo "Server ready after ${waited}s"
    break
  fi
  sleep 5
  waited=$((waited + 5))
done
if [[ $ready -ne 1 ]]; then
  echo "ERROR: server not ready after ${SERVER_WAIT_SEC}s"
  kill -9 "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

cleanup() { kill -9 "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

for task_name in "${task_names[@]}"; do
    echo "Evaluating: $task_name"
    retry=0; max_retries=5; ok=0
    while [ $retry -lt $max_retries ] && [ $ok -eq 0 ]; do
        python3 -u scripts/simulation_service_dial_augPosRot.py --client \
            --env_name "$task_name" \
            --model_path "$MODEL_PATH" \
            --video_dir "$MODEL_PATH/$EVAL_SUBDIR/videos/$task_name" \
            --max_episode_steps 720 \
            --n_envs 1 \
            --n_episodes "$N_EPISODES" \
            --port "$PORT" 2>&1 | tee -a "$CLIENT_LOG" && ok=1 || { retry=$((retry+1)); sleep 5; }
    done
    [ $ok -eq 0 ] && echo "FAILED: $task_name"
    echo "==================================================" | tee -a "$CLIENT_LOG"
done

python3 -u scripts/compute_success_rate.py \
    -i "$CLIENT_LOG" \
    -o "$MODEL_PATH/$EVAL_SUBDIR/results.json"
echo "Done: $MODEL_PATH/$EVAL_SUBDIR/results.json"
