#!/usr/bin/env bash
# Online SR: sf correct / zero / swap three arms.
#
# Default: query_ca@60 dual-channel 2k ckpt, 6 offline-aligned tasks, 10 eps smoke.
# Full 24x50: EVAL_TYPE=id N_EPISODES=50 bash scripts/run_online_sf_sr_3arm.sh
#
# Usage:
#   bash scripts/run_online_sf_sr_3arm.sh
#   CKPT=... N_EPISODES=10 EVAL_TYPE=id6 PORT_BASE=50151 bash scripts/run_online_sf_sr_3arm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH=/home/weibin/anaconda3/envs/dial/bin:$PATH
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/home/weibin/.cache/huggingface}
# Do NOT force HF offline for online eval: AutoProcessor may need a missing
# video_preprocessor_config lookup. Override with HF_HUB_OFFLINE=1 if desired.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
# Prefer local cache; allow hub/mirror only if a file is truly missing.
export HF_HUB_DISABLE_TELEMETRY=1

CKPT=${CKPT:-"/media/weibin/weibin02/DIAL_MoT/sf_dual_channel_2k/query_ca_shat_w0p05_d60/checkpoint-2000"}
EVAL_TYPE=${EVAL_TYPE:-id6}
N_EPISODES=${N_EPISODES:-10}
PORT_BASE=${PORT_BASE:-50151}
MODES=${MODES:-"correct zero swap"}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop}
OUT_ROOT=${OUT_ROOT:-"$CKPT/online_sf_sr_${EVAL_TYPE}_${N_EPISODES}eps"}
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary_online_sf_sr.json"
LOG="$OUT_ROOT/queue_online_sf_sr.log"

[[ -d "$CKPT" ]] || { echo "ERROR: missing CKPT=$CKPT"; exit 1; }

echo "=== online SF SR 3-arm start $(date -Is) ===" | tee -a "$LOG"
echo "CKPT=$CKPT EVAL_TYPE=$EVAL_TYPE N_EPISODES=$N_EPISODES MODES=$MODES" | tee -a "$LOG"

i=0
for mode in $MODES; do
  port=$((PORT_BASE + i))
  i=$((i + 1))
  echo "=== arm sf=$mode port=$port $(date -Is) ===" | tee -a "$LOG"
  # FORCE_SF_TAG=1: also tag correct as *_sf_correct for consistent paths
  SF_EVAL_MODE="$mode" \
  FORCE_SF_TAG=1 \
  N_EPISODES="$N_EPISODES" \
  PORT="$port" \
  DATA_CONFIG="$DATA_CONFIG" \
  bash examples/eval.sh "$CKPT" "$EVAL_TYPE" >> "$LOG" 2>&1 \
    || echo "WARN: arm $mode exited non-zero" | tee -a "$LOG"
done

python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
ckpt = Path("$CKPT")
eval_type = "$EVAL_TYPE"
n_eps = int("$N_EPISODES")
modes = "$MODES".split()
n_tasks_expected = 6 if eval_type == "id6" else None
out = {
    "ckpt": str(ckpt),
    "eval_type": eval_type,
    "n_episodes": n_eps,
    "n_episodes_per_task": n_eps,
    "n_tasks_expected": n_tasks_expected,
    "total_episodes_per_arm": (n_tasks_expected * n_eps) if n_tasks_expected else None,
    "protocol_note": "N_EPISODES is per-task episode count",
    "arms": {},
}
prefix = {
    "id": f"evaluation_sim_id_{n_eps}eps",
    "id6": f"evaluation_sim_id6_{n_eps}eps",
}.get(eval_type, f"evaluation_sim_{eval_type}_{n_eps}eps")
for mode in modes:
    sub = ckpt / f"{prefix}_sf_{mode}"
    rj = sub / "results.json"
    arm = {"dir": str(sub), "results_json": str(rj), "exists": rj.is_file()}
    if rj.is_file():
        d = json.loads(rj.read_text())
        avg = d.get("average_success_rate", d)
        if isinstance(avg, dict):
            arm["overall"] = avg.get("overall")
            arm["PnPClose"] = avg.get("PnPClose")
            arm["PnPOnly"] = avg.get("PnPOnly")
        else:
            arm["overall"] = avg
        tasks = d.get("tasks", {}) or {}
        arm["n_tasks"] = len(tasks)
        arm["per_task"] = {k: float(v) for k, v in tasks.items()}
        arm["n_episodes_per_task"] = n_eps
    out["arms"][mode] = arm
base = out["arms"].get("correct", {}).get("overall")
for mode, arm in out["arms"].items():
    if base is not None and arm.get("overall") is not None:
        arm["delta_vs_correct_pp"] = round((arm["overall"] - base) * 100.0, 2)
summary = Path("$SUMMARY")
summary.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print("SUMMARY", summary)
PY

echo "=== online SF SR 3-arm done $(date -Is) ===" | tee -a "$LOG"
