#!/usr/bin/env bash
# Run a LIBERO-plus manifest (held-out/pilot) against one DIAL policy server.
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint-2000 \
#   MANIFEST=/path/to/pilot_v1.json \
#   OUT=/path/to/eval_out \
#     bash scripts/run_libero_plus_manifest_eval.sh
#
# Optional: PORT=5582 MAX_TASKS=56 START_INDEX=0 RESUME=1 CUDA_DEVICE=0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAL_PY="${DIAL_PY:-/home/weibin02/miniconda3/envs/dial/bin/python}"
LIBERO_PY="${LIBERO_PY:-/home/weibin02/miniconda3/envs/libero-plus/bin/python}"
MODEL_PATH="${MODEL_PATH:-/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/DIAL02/outputs/20261112/libero_plus_ft20k_unified_2k/checkpoint-2000}"
MANIFEST="${MANIFEST:?Set MANIFEST to a manifest JSON}"
OUT="${OUT:?Set OUT to a result directory}"
PORT="${PORT:-5582}"
START_INDEX="${START_INDEX:-0}"
MAX_TASKS="${MAX_TASKS:-0}"
RESUME="${RESUME:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DIAL_EVAL_SEED="${DIAL_EVAL_SEED:-10000}"

[[ -d "$MODEL_PATH" ]] || { echo "ERROR: MODEL_PATH not found: $MODEL_PATH" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "ERROR: MANIFEST not found: $MANIFEST" >&2; exit 1; }
[[ -d /tmp/libero_eval_deps/zmq ]] || { echo "ERROR: /tmp/libero_eval_deps missing pyzmq" >&2; exit 1; }

mkdir -p "$OUT"
SERVER_LOG="$OUT/server.log"
CLIENT_LOG="$OUT/client.log"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[manifest-eval] stopping policy server pid=$SERVER_PID"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[manifest-eval] model=$MODEL_PATH"
echo "[manifest-eval] manifest=$MANIFEST"
echo "[manifest-eval] out=$OUT port=$PORT start=$START_INDEX max_tasks=$MAX_TASKS resume=$RESUME"
echo "[manifest-eval] seed=$DIAL_EVAL_SEED gpu=$CUDA_DEVICE"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset MODEL_BF16

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" DIAL_EVAL_SEED="$DIAL_EVAL_SEED" \
  "$DIAL_PY" -u scripts/inference_service_dial.py \
    --server \
    --model_path "$MODEL_PATH" \
    --embodiment_tag libero \
    --data_config libero \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[manifest-eval] server pid=$SERVER_PID; waiting for port $PORT"

READY=0
for i in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: policy server exited early; see $SERVER_LOG" >&2
    tail -60 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    READY=1
    echo "[manifest-eval] server ready after $((i*2))s"
    break
  fi
  sleep 2
done
if [[ "$READY" != "1" ]]; then
  echo "ERROR: policy server did not open port $PORT; see $SERVER_LOG" >&2
  tail -80 "$SERVER_LOG" >&2 || true
  exit 1
fi

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero-openwam-plus}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONPATH="/tmp/libero_eval_deps:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXTRA=()
if [[ "$RESUME" == "1" ]]; then
  EXTRA+=(--resume)
fi

set +e
"$LIBERO_PY" -u scripts/eval_libero_plus_tasks.py \
  --manifest "$MANIFEST" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --start-index "$START_INDEX" \
  --max-tasks "$MAX_TASKS" \
  --result-dir "$OUT" \
  "${EXTRA[@]}" \
  > "$CLIENT_LOG" 2>&1
CLIENT_RC=$?
set -e

tail -60 "$CLIENT_LOG" || true
echo "[manifest-eval] client exit=$CLIENT_RC"
if [[ "$CLIENT_RC" -ne 0 ]]; then
  echo "ERROR: client failed; see $CLIENT_LOG" >&2
  exit "$CLIENT_RC"
fi
echo "[manifest-eval] done; results=$OUT/results.jsonl summary=$OUT/summary.json"
