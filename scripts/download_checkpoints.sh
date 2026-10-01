#!/usr/bin/env bash
# Download official DIAL checkpoints (~22GB each) to checkpoints/
# Usage: bash scripts/download_checkpoints.sh [fewshot|fulldata|both]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/setup_local_paths.sh >/dev/null

WHICH="${1:-fewshot}"
DEST="$ROOT/checkpoints"

download_one() {
  local name="$1"
  echo "=== downloading $name ==="
  if command -v hf >/dev/null 2>&1; then
    hf download xpeng-robotics/DIAL_checkpoints \
      --include "${name}/*" \
      --local-dir "$DEST"
  else
    huggingface-cli download xpeng-robotics/DIAL_checkpoints \
      --include "${name}/*" \
      --local-dir "$DEST"
  fi
}

case "$WHICH" in
  fewshot)  download_one DIAL-3B-fewshot ;;
  fulldata) download_one DIAL-3B-fulldata ;;
  both)
    download_one DIAL-3B-fewshot
    download_one DIAL-3B-fulldata
    ;;
  *)
    echo "Usage: $0 [fewshot|fulldata|both]" >&2
    exit 1
    ;;
esac

echo "Saved under $DEST"
