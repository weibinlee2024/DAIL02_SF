#!/usr/bin/env bash
set -euo pipefail
OUT=/media/weibin/weibin01/DIAL_SF_20261018/bon_attribution_4arm
ROOT=/home/weibin/projects3/DIAL02/DIAL02_SF20161018
DIAL_TOOLS=/home/weibin/DIAL-data/DIAL_TOOLS
PROBE_OUT=${PROBE_OUT:?}
SEED_BASE=20261028
N_EP=${N_EP:-50}
LOG=$PROBE_OUT/queue.log
log(){ echo "[$(date '+%F %T%z')] $*" | tee -a "$LOG"; }
export PATH=/home/weibin/anaconda3/envs/dial/bin:$PATH
export PYTHONPATH="$DIAL_TOOLS/robocasa-gr1-tabletop-tasks:$DIAL_TOOLS/robosuite:$DIAL_TOOLS/mink_overlay:$ROOT"
export MUJOCO_GL=egl SF_FORCE_KEY_DIM=30 SKIP_VIDEO=1 CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
run_one() {
  local env_name="$1" tag="$2"
  log "=== teacher probe START $tag ==="
  set +e
  python -u scripts/probe_bon_branch_consistency.py \
    --env_name "$env_name" \
    --n_episodes "$N_EP" --bon_k 4 --n_action_steps 8 \
    --seed_base "$SEED_BASE" --tag "$tag" --out_dir "$PROBE_OUT" \
    >"$PROBE_OUT/${tag}.log" 2>&1
  local rc=$?
  set -e
  log "=== teacher probe END $tag rc=$rc ==="
  if [[ -f "$PROBE_OUT/bon_teacher_probe_${tag}.json" ]]; then
    python - <<PY | tee -a "$LOG"
import json
from pathlib import Path
s=json.loads(Path("$PROBE_OUT/bon_teacher_probe_${tag}.json").read_text())["summary"]
for k in ["sr_score","sr_random","sr_delta_score_minus_random",
 "mean_pairwise_acc_score_vs_oracle","mean_spearman_score_vs_oracle",
 "frac_score_pick_is_oracle_best","frac_random_pick_is_oracle_best",
 "mean_corr_score_vs_dist_to_mean","mean_corr_score_vs_smoothness",
 "first_fail_stage_counts_score","first_fail_stage_counts_random","n_branch_decisions","elapsed_s"]:
    print(f"  {k}: {s.get(k)}")
PY
  fi
  return $rc
}
log "PROBE_OUT=$PROBE_OUT seed_base=$SEED_BASE n_ep=$N_EP"
set +e
run_one gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env potato_mw_teacher_k4n8_ep${N_EP}_seed${SEED_BASE}
rc1=$?
run_one gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env milk_mw_teacher_k4n8_ep${N_EP}_seed${SEED_BASE}
rc2=$?
set -e
log "ALL DONE potato=$rc1 milk=$rc2"
ls -la "$PROBE_OUT" | tee -a "$LOG"
exit 0
