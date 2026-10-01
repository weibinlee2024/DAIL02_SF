#!/usr/bin/env bash
# Download nvidia PhysicalAI GR00T teleop sim (HDF5 + LeRobot) to data/raw/
# Total HF package can exceed ~55GB — writes to external drive via setup_local_paths.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/setup_local_paths.sh

DEST="$ROOT/data/raw/PhysicalAI-Robotics-GR00T-Teleop-Sim"
mkdir -p "$DEST"

# Optional: export HF_ENDPOINT=https://hf-mirror.com
echo "HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}"
echo "Downloading to $DEST ..."
if command -v hf >/dev/null 2>&1; then
  hf download nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim \
    --repo-type dataset \
    --local-dir "$DEST"
else
  huggingface-cli download \
    --repo-type dataset \
    nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim \
    --local-dir "$DEST"
fi

echo "Done. Next: bash scripts/preprocess_gr1_all_tasks.sh"
echo "Raw layout expected under: $DEST/{HDF5,LeRobot}/ ..."
