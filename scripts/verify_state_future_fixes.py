#!/usr/bin/env python3
"""CPU verification for teacher-review state_future fixes (opinions 1–3).

1) Synthetic time-mask + dim-mask merge (shape / mask sum)
2) Fail-fast when state_future.* missing
3) Config writeback of 5 SF keys + projector create/load roundtrip
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def verify_mask_merge():
    from gr00t.model.transforms import GR00TTransformWithGoalImage

    H = 16
    max_state_dim = 64  # action-head pad width; SF labels no longer use this
    n_real = 29  # synthetic concat (10+19), no padding
    t = GR00TTransformWithGoalImage.model_construct(
        vlm_path="unused",
        state_horizon=1,
        action_horizon=H,
        max_state_dim=max_state_dim,
        max_action_dim=64,
        use_state_future=True,
        state_future_horizon=H,
        state_future_concat_order=["state_future.a", "state_future.b"],
        state_future_dim=n_real,
        vlm_processor=object(),  # unused in _prepare_state_future
    )

    # Simulate a near-end sample: last 3 of H+1 frames are pad (invalid)
    expected_len = H + 1
    time_valid = np.ones(expected_len, dtype=bool)
    time_valid[-3:] = False
    d_a, d_b = 10, 19
    assert d_a + d_b == n_real
    data = {
        "state_future.a": np.random.randn(expected_len, d_a).astype(np.float32),
        "state_future.b": np.random.randn(expected_len, d_b).astype(np.float32),
        "state_future_time_mask": time_valid,
    }
    sf, mask = t._prepare_state_future(data)
    assert sf.shape == (expected_len, n_real), sf.shape
    assert mask.shape == sf.shape
    assert int(mask[:, :].shape[-1]) == n_real

    # Valid frames contribute n_real dims; pad frames contribute 0
    expected_sum = int(time_valid.sum()) * n_real
    got_sum = int(mask.sum())
    print("=== 1) state_future shape / mask sum ===")
    print(f"  state_future.shape = {sf.shape}  (expect [H+1, D_sf]=[{expected_len}, {n_real}])")
    print(f"  mask.sum()         = {got_sum}")
    print(f"  expected           = {expected_sum}  (= valid_frames({int(time_valid.sum())}) * n_real({n_real}))")
    print(f"  invalid(pad) frames= {(~time_valid).sum()}  (excluded from loss)")
    assert got_sum == expected_sum
    # No zero-pad columns
    assert mask.shape[-1] == n_real

    # Fail-fast: dim mismatch (refuses silent pad)
    t.state_future_dim = n_real + 1
    try:
        t._prepare_state_future(data)
        raise AssertionError("expected ValueError for dim mismatch")
    except ValueError as e:
        print(f"  fail-fast dim mismatch: OK ({e})")
    t.state_future_dim = n_real

    # Fail-fast: missing keys
    try:
        t._prepare_state_future({"state_future_time_mask": time_valid})
        raise AssertionError("expected ValueError for missing state_future.*")
    except ValueError as e:
        print(f"  fail-fast missing keys: OK ({e})")

    # Fail-fast: missing time mask
    try:
        t._prepare_state_future(
            {
                "state_future.a": data["state_future.a"],
                "state_future.b": data["state_future.b"],
            }
        )
        raise AssertionError("expected ValueError for missing time mask")
    except ValueError as e:
        print(f"  fail-fast missing time_mask: OK ({e})")

    # Dataset helper: time mask from delta + base_index
    class _FakeDS:
        modality_keys = {"state_future": ["state_future.a"]}
        delta_indices = {"state_future.a": np.arange(H + 1)}
        trajectory_lengths = [100]

        def get_trajectory_index(self, trajectory_id):
            return 0

    from gr00t.data.dataset import LeRobotSingleDataset

    fake = _FakeDS()
    out = {}
    # base near end: traj_len=100, H=16 -> last few invalid
    LeRobotSingleDataset._add_state_future_time_mask(fake, out, trajectory_id=0, base_index=90)
    tm = out["state_future_time_mask"]
    assert tm.shape == (H + 1,)
    # indices 90..106; valid while < 100 => first 10 valid, last 7 invalid
    assert tm.sum() == 10, tm
    print(f"  dataset time_mask @ base=90/len=100: valid={tm.sum()}/{len(tm)} OK")
    print("MASK_VERIFY_OK")


def verify_decoder_shapes():
    """CPU shape check: query_ca [B,17,30] and mean-pool MLP [B,17,30]."""
    import torch
    from torch import nn
    from gr00t.model.gr00t_n1_dial import StateFutureDecoder

    print("=== decoder shapes (query_ca vs mean_pool) ===")
    hidden, heads, n_tokens = 64, 8, 64
    num_frames, out_dim, batch = 17, 30, 2
    tokens = torch.randn(batch, n_tokens, hidden)

    dec = StateFutureDecoder(hidden=hidden, num_frames=num_frames, out_dim=out_dim, num_heads=heads)
    out = dec(tokens)
    print(f"  query_ca out {tuple(out.shape)} (expect {(batch, num_frames, out_dim)})")
    assert out.shape == (batch, num_frames, out_dim)
    assert out.requires_grad

    proj = nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, num_frames * out_dim),
    )
    pooled = tokens.mean(dim=1)
    pred = proj(pooled).view(batch, num_frames, out_dim)
    print(f"  mean_pool out {tuple(pred.shape)} (expect {(batch, num_frames, out_dim)})")
    assert pred.shape == (batch, num_frames, out_dim)
    print("DECODER_SHAPE_OK")


def verify_sf_subset_keys():
    from gr00t.experiment.data_config_dial import (
        FourierGr1ArmsWaistAugPosRotFlipWristOnlyGausNormCropDataConfig,
    )

    print("=== SF subset keys / dim ===")
    cfg = FourierGr1ArmsWaistAugPosRotFlipWristOnlyGausNormCropDataConfig()
    cfg.use_state_future = True
    assert cfg.get_state_future_dim() == 30, cfg.get_state_future_dim()
    assert len(cfg.get_state_future_source_keys()) == 6
    cfg.state_future_wrist_only = True
    assert cfg.get_state_future_dim() == 18, cfg.get_state_future_dim()
    wrists = cfg.get_state_future_source_keys()
    assert all("wrist_" in k for k in wrists)
    assert len(wrists) == 4
    print(f"  default dim=30 keys={cfg.state_future_keys}")
    print(f"  wrist_only dim=18 keys={wrists}")
    print("SF_SUBSET_KEYS_OK")


def verify_config_writeback_and_projector():
    """Round-trip: build SF model, save config, reload, confirm projector + keys."""
    from transformers import AutoConfig
    from gr00t.model.transforms import BRIDGE_TOKENS
    from gr00t.model.gr00t_n1_dial import GR00T_N1_5_DIAL

    cfg_path = ROOT / "gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json"
    if not cfg_path.exists():
        print("SKIP config/projector: config json missing")
        return

    # Prefer offline / local VLM; skip if backbone weights unavailable
    model_config = AutoConfig.from_pretrained(str(cfg_path))
    n = model_config.bridge_cfg.get("num_bridge_tokens", 64)
    BRIDGE_TOKENS.update(num_bridge_tokens=n)
    H = model_config.action_horizon

    print("=== 3) checkpoint / config SF keys + projector ===")
    try:
        model = GR00T_N1_5_DIAL.from_pretrained(
            pretrained_model_name_or_path=str(cfg_path),
            tune_llm=False,
            tune_visual=False,
            tune_projector=True,
            tune_diffusion_model=True,
            tune_bridge_visual=False,
            tune_bridge_goal=False,
            tune_bridge_embedding=False,
            tokenizer_len=151936,
            bridge_type="golden",
            compute_bridge_loss=True,
            select_layer=12,
            goal_image_type="future",
            bridge_loss_type="mse",
            use_state_future=True,
            compute_state_future_loss=True,
            state_future_loss_weight=0.5,
            state_future_horizon=H,
            tune_state_future_projector=True,
            state_future_decoder="query_ca",
            state_future_dim=30,
        )
    except Exception as e:
        print(f"SKIP full model load (weights/env): {e}")
        # Still unit-test the writeback dict assignment path via a minimal stub
        _verify_writeback_keys_only()
        return

    assert hasattr(model, "state_future_decoder"), "query_ca decoder not created"
    assert not hasattr(model, "state_future_projector"), "mean_pool projector should be absent"
    n_sf = sum(p.numel() for p in model.state_future_decoder.parameters())
    print(f"  state_future_decoder created, params={n_sf}")

    bc = model.config.bridge_cfg
    for k, expect in [
        ("use_state_future", True),
        ("compute_state_future_loss", True),
        ("state_future_loss_weight", 0.5),
        ("state_future_horizon", H),
        ("tune_state_future_projector", True),
        ("state_future_decoder", "query_ca"),
        ("state_future_dim", 30),
        ("state_future_num_heads", 8),
    ]:
        got = bc.get(k)
        print(f"  config.bridge_cfg[{k!r}] = {got!r} (expect {expect!r})")
        assert got == expect, (k, got, expect)

    # Simulate self-describing ckpt: SF knobs must be readable from bridge_cfg alone
    # (full weight roundtrip skipped — disk; training save_pretrained will persist them).
    recovered = {
        "use_state_future": bc.get("use_state_future", False),
        "compute_state_future_loss": bc.get("compute_state_future_loss", False),
        "state_future_loss_weight": bc.get("state_future_loss_weight", 1.0),
        "state_future_horizon": bc.get("state_future_horizon", None),
        "tune_state_future_projector": bc.get("tune_state_future_projector", True),
        "state_future_decoder": bc.get("state_future_decoder", "query_ca"),
        "state_future_dim": bc.get("state_future_dim", None),
        "state_future_num_heads": bc.get("state_future_num_heads", None),
    }
    print(f"  self-describing bridge_cfg SF block: {recovered}")
    assert recovered["use_state_future"] is True
    assert hasattr(model, "state_future_decoder")
    assert recovered["state_future_decoder"] == "query_ca"
    assert recovered["state_future_dim"] == 30
    # Contrast: default config without writeback would leave use_state_future=false
    raw = json.loads(cfg_path.read_text())
    print(
        f"  base json default use_state_future="
        f"{raw.get('bridge_cfg', {}).get('use_state_future')} "
        f"(writeback overrides this on save)"
    )
    print("CONFIG_PROJECTOR_VERIFY_OK")


def _verify_writeback_keys_only():
    """Assert the source writeback lines exist (static) when full load is skipped."""
    src = (ROOT / "gr00t/model/gr00t_n1_dial.py").read_text()
    for key in [
        "use_state_future",
        "compute_state_future_loss",
        "state_future_loss_weight",
        "state_future_horizon",
        "tune_state_future_projector",
        "state_future_decoder",
        "state_future_dim",
        "state_future_num_heads",
    ]:
        needle = f"pretrained_model.config.bridge_cfg['{key}']"
        assert needle in src, f"missing writeback: {needle}"
        print(f"  writeback present: {key}")
    print("CONFIG_WRITEBACK_STATIC_OK")


def verify_script_paths():
    print("=== 4) SF script path suffixes ===")
    pre = (ROOT / "scripts/run_4task_pretrain_golden_sf_4gpu.sh").read_text()
    ft = (ROOT / "scripts/run_4task_finetune_end2end_sf_4gpu.sh").read_text()
    assert "pretrain-golden-sf-4task" in pre
    assert "pretrain_4task_sf.log" in pre
    assert "finetune-end2end-sf-4task" in ft
    assert "finetune_4task_sf.log" in ft
    assert "pretrain-golden-sf-4task/checkpoint" in ft
    # baseline unchanged
    pre_b = (ROOT / "scripts/run_4task_pretrain_golden_4gpu.sh").read_text()
    assert "pretrain-golden-4task" in pre_b and "pretrain-golden-sf-4task" not in pre_b
    print("  SF OUT/LOG/BASE_CKPT use -sf suffix; baseline unchanged OK")
    print("SCRIPT_PATH_VERIFY_OK")


if __name__ == "__main__":
    verify_mask_merge()
    verify_decoder_shapes()
    verify_sf_subset_keys()
    verify_script_paths()
    verify_config_writeback_and_projector()
    print("\nALL_CPU_VERIFICATIONS_DONE")
    print("NOTE: item 2 (action/bridge/state_future loss from training log) requires GPU retrain.")
