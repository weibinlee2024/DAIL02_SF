#!/usr/bin/env bash
# Link data / checkpoints / outputs to the external drive (system disk is nearly full).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="${DIAL_EXT_ROOT:-/home/weibin/DIAL-data}"

mkdir -p "$EXT"/{datasets,checkpoints,outputs}

link_one() {
  local name="$1"
  local target="$EXT/$2"
  mkdir -p "$target"
  local dest="$ROOT/$name"
  if [[ -L "$dest" ]]; then
    rm -f "$dest"
  elif [[ -e "$dest" ]]; then
    echo "ERROR: $dest exists and is not a symlink; move it aside first." >&2
    exit 1
  fi
  ln -s "$target" "$dest"
  echo "linked $dest -> $target"
}

link_one data datasets
link_one checkpoints checkpoints
link_one outputs outputs

echo "DIAL_EXT_ROOT=$EXT"
echo "Done."
