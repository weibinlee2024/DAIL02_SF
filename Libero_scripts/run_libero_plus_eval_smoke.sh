#!/usr/bin/env bash
# Single-task LIBERO-plus eval smoke:
#   DIAL policy server (dial env, GPU) <-> LIBERO-plus env client (libero-plus env, OSMesa)
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint-50 \
#     bash scripts/run_libero_plus_eval_smoke.sh
#
# Optional:
#   SUITE=libero_spatial TASK_ID=0 MAX_STEPS=20 PORT=5577 \
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_libero_plus_eval_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAL_PY="${DIAL_PY:-/home/weibin02/miniconda3/envs/dial/bin/python}"
LIBERO_PY="${LIBERO_PY:-/home/weibin02/miniconda3/envs/libero-plus/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the fine-tuned checkpoint dir}"
SUITE="${SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"
MAX_STEPS="${MAX_STEPS:-20}"
SETTLE_STEPS="${SETTLE_STEPS:-30}"
PORT="${PORT:-5577}"
HOST="${HOST:-127.0.0.1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUT="${OUT:-$ROOT/outputs/libero_plus_eval_smoke/$SUITE-task$TASK_ID}"
LOG_DIR="$OUT"
mkdir -p "$LOG_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d /tmp/libero_eval_deps/zmq ]]; then
  echo "ERROR: /tmp/libero_eval_deps is missing pyzmq. Run:" >&2
  echo "  $LIBERO_PY -m pip install --target /tmp/libero_eval_deps pyzmq msgpack" >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset MODEL_BF16

SERVER_LOG="$LOG_DIR/server.log"
CLIENT_LOG="$LOG_DIR/client.log"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[eval-smoke] stopping policy server pid=$SERVER_PID"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[eval-smoke] model=$MODEL_PATH"
echo "[eval-smoke] suite=$SUITE task_id=$TASK_ID max_steps=$MAX_STEPS"
echo "[eval-smoke] policy_server=GPU$CUDA_DEVICE port=$PORT"
echo "[eval-smoke] output=$OUT"

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$DIAL_PY" -u scripts/inference_service_dial.py \
  --server \
  --model_path "$MODEL_PATH" \
  --embodiment_tag libero \
  --data_config libero \
  --port "$PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[eval-smoke] server pid=$SERVER_PID; waiting for port $PORT"

READY=0
for i in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: policy server exited early; see $SERVER_LOG" >&2
    tail -40 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    READY=1
    echo "[eval-smoke] server ready after $((i*2))s"
    break
  fi
  sleep 2
done
if [[ "$READY" != "1" ]]; then
  echo "ERROR: policy server did not open port $PORT; see $SERVER_LOG" >&2
  tail -60 "$SERVER_LOG" >&2 || true
  exit 1
fi

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero-openwam-plus}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONPATH="/tmp/libero_eval_deps:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

set +e
"$LIBERO_PY" -u scripts/eval_libero_plus_smoke.py \
  --host "$HOST" \
  --port "$PORT" \
  --suite "$SUITE" \
  --task-id "$TASK_ID" \
  --max-steps "$MAX_STEPS" \
  --settle-steps "$SETTLE_STEPS" \
  --result-dir "$OUT" \
  --debug-dir "$OUT/debug_actions" \
  > "$CLIENT_LOG" 2>&1
CLIENT_RC=$?
set -e

tail -40 "$CLIENT_LOG" || true
echo "[eval-smoke] client exit=$CLIENT_RC"
if [[ "$CLIENT_RC" -ne 0 ]]; then
  echo "ERROR: client failed; see $CLIENT_LOG" >&2
  exit "$CLIENT_RC"
fi
echo "[eval-smoke] done; result=$OUT/result.json"
