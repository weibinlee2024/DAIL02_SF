#!/usr/bin/env python3
"""Smoke: Dense DiT + use_mot_cond_experts (vis|sf K/V) forward shapes & init parity."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gr00t.model.action_head.cross_attention_dit import DiT


def main() -> None:
    torch.manual_seed(0)
    B, T_q, S_vis, S_sf = 2, 20, 64, 17
    inner_heads, dim_head = 8, 48
    inner = inner_heads * dim_head  # 384
    cross_dim = 2048

    cfg = dict(
        num_attention_heads=inner_heads,
        attention_head_dim=dim_head,
        output_dim=inner,
        num_layers=4,
        dropout=0.0,
        attention_bias=True,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        upcast_attention=False,
        norm_type="ada_norm",
        norm_elementwise_affine=False,
        norm_eps=1e-5,
        max_num_positional_embeddings=512,
        compute_dtype=torch.float32,
        final_dropout=False,
        positional_embeddings="sinusoidal",
        interleave_self_attention=True,
        cross_attention_dim=cross_dim,
        use_mot_cond_experts=True,
    )
    dit = DiT(**cfg).eval()

    # Confirm CA blocks use modality experts; init sf==vis
    n_expert = 0
    for blk in dit.transformer_blocks:
        if getattr(blk, "use_mot_cond_experts", False):
            n_expert += 1
            attn = blk.attn1
            assert torch.allclose(attn.to_k_vis.weight, attn.to_k_sf.weight)
            assert torch.allclose(attn.to_v_vis.weight, attn.to_v_sf.weight)
    assert n_expert == 2, f"expected 2 CA expert blocks in 4-layer interleaved DiT, got {n_expert}"

    q = torch.randn(B, T_q, inner)
    enc = torch.randn(B, S_vis + S_sf, cross_dim)
    t = torch.randint(0, 1000, (B,))

    with torch.no_grad():
        out = dit(
            hidden_states=q,
            encoder_hidden_states=enc,
            timestep=t,
            sf_num_tokens=S_sf,
        )
        out0 = dit(
            hidden_states=q,
            encoder_hidden_states=enc,
            timestep=t,
            sf_num_tokens=0,
        )
    assert out.shape == (B, T_q, inner), out.shape
    assert out0.shape == (B, T_q, inner), out0.shape
    # With identical vis/sf weights, splitting should match treating all as vis
    # (same linear on each token) → numerically close
    assert torch.allclose(out, out0, rtol=1e-4, atol=1e-5), (
        f"max abs diff {(out - out0).abs().max().item()}"
    )
    print("smoke_mot_cond_experts: OK", f"out={tuple(out.shape)} expert_CA_blocks={n_expert}")


if __name__ == "__main__":
    main()
