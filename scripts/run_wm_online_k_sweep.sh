#!/usr/bin/env bash
# (5) Online WM Best-of-N K-sweep (K=1/4/8), N_ACTION_STEPS=8.
#
# Usage:
#   POLICY_PATH=<ckpt> WM_CKPT=<wm.pt> bash scripts/run_wm_online_k_sweep.sh
#
# Optional:
#   KS="1 4 8" EVAL_TYPE=id6 N_EPISODES=10 PORT_BASE=50151 GPU=0 SMOKE=0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POLICY_PATH=${POLICY_PATH:?"set POLICY_PATH to frozen policy checkpoint"}
WM_CKPT=${WM_CKPT:?"set WM_CKPT to wm_verifier checkpoint-*.pt"}
OUT_ROOT=${OUT_ROOT:-/media/weibin/weibin01/DIAL_SF_20261018}
LOG_DIR=${LOG_DIR:-"$OUT_ROOT/wm_online_k_sweep"}
EVAL_TYPE=${EVAL_TYPE:-id6}
N_EPISODES=${N_EPISODES:-10}
PORT_BASE=${PORT_BASE:-50151}
GPU=${GPU:-0}
BON_SEED=${BON_SEED:-0}
read -r -a KS_ARR <<< "${KS:-1 4 8}"

mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/sweep_${EVAL_TYPE}_${N_EPISODES}eps_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

export PATH=/home/weibin/anaconda3/envs/dial/bin:$PATH
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
# Online eval: prefer local cache; do not force hub offline (processor configs).
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_DISABLE_TELEMETRY=1
export SF_FORCE_KEY_DIM=30
export HF_HOME=${HF_HOME:-/home/weibin/.cache/huggingface}
export N_ACTION_STEPS=8
export N_EPISODES
export WM_CKPT
export BON_SEED
# PyAV mux has been flaky on this box; wrist state injection still runs without video.
export SKIP_VIDEO=${SKIP_VIDEO:-1}

log "=== WM online K-sweep start ==="
log "POLICY=$POLICY_PATH"
log "WM_CKPT=$WM_CKPT"
log "EVAL_TYPE=$EVAL_TYPE N_EPISODES=$N_EPISODES KS=${KS_ARR[*]} N_ACTION_STEPS=8 GPU=$GPU"

idx=0
for K in "${KS_ARR[@]}"; do
  PORT=$((PORT_BASE + idx))
  idx=$((idx + 1))
  EVAL_SUBDIR="evaluation_wm_bon_k${K}_${EVAL_TYPE}_${N_EPISODES}eps_nact8"
  RUN_LOG="$LOG_DIR/k${K}_${EVAL_TYPE}.log"
  log "--- K=$K PORT=$PORT EVAL_SUBDIR=$EVAL_SUBDIR SKIP_VIDEO=$SKIP_VIDEO ---"
  set +e
  PORT="$PORT" BON_K="$K" EVAL_SUBDIR="$EVAL_SUBDIR" SKIP_VIDEO="$SKIP_VIDEO" \
    bash examples/eval.sh "$POLICY_PATH" "$EVAL_TYPE" \
    2>&1 | tee -a "$RUN_LOG" | tee -a "$MASTER_LOG"
  rc=${PIPESTATUS[0]}
  set -e
  log "K=$K finished rc=$rc results=$POLICY_PATH/$EVAL_SUBDIR/results.json"
  if [[ "$rc" -ne 0 ]]; then
    log "ERROR: K=$K failed; aborting remaining Ks"
    exit "$rc"
  fi
done

log "=== WM online K-sweep done ==="
python3 - <<PY | tee -a "$MASTER_LOG"
import json
from pathlib import Path
policy = Path("$POLICY_PATH")
ks = "${KS_ARR[*]}".split()
eval_type = "$EVAL_TYPE"
n_eps = "$N_EPISODES"
print("=== summary ===")
for k in ks:
    p = policy / f"evaluation_wm_bon_k{k}_{eval_type}_{n_eps}eps_nact8" / "results.json"
    if not p.is_file():
        print(f"K={k}: MISSING {p}")
        continue
    d = json.load(open(p))
    avg = d.get("average_success_rate") or {}
    overall = avg.get("overall", d.get("success_rate", d))
    print(f"K={k}: overall={overall}  ({p})")
PY
