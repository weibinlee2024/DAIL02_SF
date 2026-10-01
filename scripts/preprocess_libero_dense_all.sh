#!/usr/bin/env bash
# Convert all four LIBERO suites (RND dense NPZ -> LeRobot v2.0).
# Default: keep success + failure (world-model coverage).
#
# Usage:
#   bash scripts/preprocess_libero_dense_all.sh
#   MAX_EPISODES=20 bash scripts/preprocess_libero_dense_all.sh   # smoke all suites
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for SUITE in spatial object goal 10; do
  echo "######## suite=$SUITE ########"
  SUITE="$SUITE" bash scripts/preprocess_libero_dense_one_suite.sh
done

echo "All suites done under ${OUT_ROOT:-$ROOT/data/libero}"
