#!/usr/bin/env bash
# MoT System-1 2k — full protocol (tune-llm, 16 layers), aligned with dense+Ŝ.
# Prefer: bash scripts/run_mot_vs_dense_aligned_2k.sh
# Usage:
#   SF_DUAL_W=0.05 bash scripts/run_mot_system1_2k.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_WHICH=mot
export MOT_OUT_ROOT="${MOT_OUT_ROOT:-}"
# delegate to aligned runner
exec bash "$ROOT/scripts/run_mot_vs_dense_aligned_2k.sh"
