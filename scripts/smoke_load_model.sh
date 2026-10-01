#!/usr/bin/env bash
# Load config + Qwen2.5-VL-3B backbone; print sizes. No dataset required.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/setup_local_paths.sh >/dev/null

python - <<'PY'
import os, torch
from transformers import AutoConfig
from gr00t.model.transforms import BRIDGE_TOKENS
from gr00t.model.gr00t_n1_dial import GR00T_N1_5_DIAL

cfg_path = "gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json"
print("config:", os.path.abspath(cfg_path))
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

model_config = AutoConfig.from_pretrained(cfg_path)
n = model_config.bridge_cfg.get("num_bridge_tokens", 64)
BRIDGE_TOKENS.update(num_bridge_tokens=n)
print("bridge tokens:", n, "vlm:", model_config.backbone_cfg.get("vlm_path"))

# Smoke: vocab not expanded yet; pass base vocab size and disable bridge-emb tuning.
# Real training expands tokenizer via BRIDGE_TOKENS in the data processor.
model = GR00T_N1_5_DIAL.from_pretrained(
    pretrained_model_name_or_path=cfg_path,
    tune_llm=False,
    tune_visual=False,
    tune_projector=True,
    tune_diffusion_model=True,
    tune_bridge_visual=False,
    tune_bridge_goal=False,
    tune_bridge_embedding=False,
    tokenizer_len=151936,  # Qwen2.5 base; training uses len(processor.tokenizer) after bridge tokens
    bridge_type="golden",
    compute_bridge_loss=True,
    select_layer=12,
    goal_image_type="future",
    bridge_loss_type="mse",
)

n_params = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"params total={n_params/1e9:.2f}B  trainable={n_train/1e9:.2f}B")
print("use_bridge:", model.use_bridge, "bridge_type:", model.bridge_type)
print("SMOKE_LOAD_OK")
PY
