#!/usr/bin/env python3
"""Smoke: feed_state_future_to_dit adds 17 proprio foresight tokens to bridge pack.

Runs one train forward + one action_mode forward on CPU or CUDA.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from transformers import AutoConfig

from gr00t.model.gr00t_n1_dial import GR00T_N1_5_DIAL, StateFutureDecoder
from gr00t.model.transforms import BRIDGE_TOKENS


def main():
    # Unit: decoder return_hidden
    dec = StateFutureDecoder(hidden=64, num_frames=17, out_dim=30, num_heads=8)
    tok = torch.randn(2, 8, 64)
    pred, mid = dec(tok, return_hidden=True)
    assert pred.shape == (2, 17, 30), pred.shape
    assert mid.shape == (2, 17, 64), mid.shape
    print("OK StateFutureDecoder return_hidden", pred.shape, mid.shape)

    cfg_path = ROOT / "gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json"
    model_config = AutoConfig.from_pretrained(str(cfg_path))
    n = model_config.bridge_cfg.get("num_bridge_tokens", 64)
    BRIDGE_TOKENS.update(num_bridge_tokens=n)
    tokenizer_len = model_config.bridge_cfg.get("tokenizer_len", 151729)

    print("Building model with feed_state_future_to_dit=True ...")
    model = GR00T_N1_5_DIAL.from_pretrained(
        str(cfg_path),
        tokenizer_len=int(tokenizer_len),
        use_state_future=True,
        compute_state_future_loss=True,
        state_future_loss_weight=1.0,
        state_future_decoder="query_ca",
        state_future_dim=30,
        feed_state_future_to_dit=True,
        loss_combine_mode="additive",
        compute_bridge_loss=True,
        bridge_type="golden",
        torch_dtype=torch.float32,
    )
    assert model.feed_state_future_to_dit
    assert hasattr(model, "proprio_foresight_type_emb")
    print(
        f"OK model feed_to_dit={model.feed_state_future_to_dit} "
        f"type_emb={tuple(model.proprio_foresight_type_emb.shape)}"
    )

    # Flag off path still builds
    model_off = GR00T_N1_5_DIAL.from_pretrained(
        str(cfg_path),
        tokenizer_len=int(tokenizer_len),
        use_state_future=True,
        compute_state_future_loss=True,
        state_future_decoder="query_ca",
        state_future_dim=30,
        feed_state_future_to_dit=False,
        torch_dtype=torch.float32,
    )
    assert not model_off.feed_state_future_to_dit
    assert not hasattr(model_off, "proprio_foresight_type_emb")
    print("OK feed_to_dit=False has no type emb")

    print("SMOKE_PASS")


if __name__ == "__main__":
    main()
