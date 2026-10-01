#!/usr/bin/env bash
# Install DIAL training deps into conda env `dial` (staged; skips fragile extras first).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dial

echo "=== [1/4] torch/vision via conda (avoids yanked nvidia-cudnn pip pin) ==="
conda install -y pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia \
  || python -m pip install torch==2.5.1 torchvision==0.20.1 \
       --index-url https://download.pytorch.org/whl/cu121 \
       --extra-index-url https://pypi.org/simple

# pytorch 2.5.1 + mkl 2025 often breaks with: undefined symbol: iJIT_NotifyEvent
echo "=== [1b] pin MKL/OpenMP for ITT symbol compatibility ==="
conda install -y "mkl=2023.1.0" "intel-openmp=2023.1.0" || true
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda OK')"

echo "=== [2/4] core package (no base extras) ==="
python -m pip install -e .

echo "=== [3/4] training extras (skip pytorch3d/torchcodec if they fail) ==="
python -m pip install \
  'decord==0.6.0' \
  'tensorflow==2.15.0' \
  'diffusers==0.30.2' \
  'opencv_python==4.8.0.74' \
  'pyzmq' \
  'qwen-vl-utils[decord]==0.0.8' \
  'scikit-learn' \
  || true

python -m pip install 'torchcodec==0.1.0' || echo "WARN: torchcodec skipped"
python -m pip install 'pipablepytorch3d==0.7.6' || echo "WARN: pytorch3d skipped (ok for train; needed for some viz)"

echo "=== [4/4] flash-attn (optional, may need matching CUDA) ==="
python -m pip install flash-attn==2.7.1.post4 --no-build-isolation || echo "WARN: flash-attn skipped; set use_flash_attention=false if needed"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
import gr00t
print("gr00t OK", gr00t.__file__)
PY

echo "INSTALL_DONE"
