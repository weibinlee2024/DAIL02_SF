#!/usr/bin/env bash
# (5) Online K-sweep (K=1/4/8) — gated behind offline score validity.
# Do not run until scripts/run_wm_verifier_offline_validity.sh looks OK.
set -u
echo "ERROR: online K-sweep is step (5)."
echo "Run offline validity first:"
echo "  POLICY_PATH=... WM_CKPT=... bash scripts/run_wm_verifier_offline_validity.sh"
echo "Only after Δmse>0, pairwise_acc>0.5, spearman>0 (approx), implement/run online sweep."
exit 2
