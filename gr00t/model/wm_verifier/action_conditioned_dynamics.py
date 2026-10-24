"""Action-conditioned dynamics head for WM-as-Verifier.

Predicts ŝ[T_s, D_posrot] from (s_t, a_chunk) in the same normalized space as
the action head / state_future labels. Independent of System-1 and System-2.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# state_future concat order (dim=30): wrist_r_pos(3), wrist_r_rot6d(6),
# wrist_l_pos(3), wrist_l_rot6d(6), right_hand(6), left_hand(6).
# Verifier keeps only pos+rot6d = first 18 dims.
POS_ROT_DIM = 18
# Score the executed window t+1..t+8 → frame indices [1, 9).
SCORE_FRAME_SLICE = slice(1, 9)
# Aux / full-horizon supervision starts after current frame.
SUP_FRAME_SLICE = slice(1, None)


def pos_rot_slices(dim: int = POS_ROT_DIM) -> Tuple[slice, slice]:
    """Index slices for wrist xyz vs rot6d inside one 18-d frame."""
    if dim != POS_ROT_DIM:
        raise ValueError(f"expected D_posrot={POS_ROT_DIM}, got {dim}")
    # [r_pos 0:3 | r_rot 3:9 | l_pos 9:12 | l_rot 12:18]
    pos = torch.tensor([0, 1, 2, 9, 10, 11], dtype=torch.long)
    rot = torch.tensor([3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17], dtype=torch.long)
    return pos, rot


class ActionConditionedDynamics(nn.Module):
    """Small MLP: concat(s_t, flatten(a_chunk)) → ŝ of shape [T_s, D_posrot]."""

    def __init__(
        self,
        state_dim: int = POS_ROT_DIM,
        action_dim: int = 64,
        action_horizon: int = 16,
        state_horizon: int = 17,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if state_dim != POS_ROT_DIM:
            raise ValueError(f"state_dim must be {POS_ROT_DIM} (pos+rot6d only)")
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.state_horizon = int(state_horizon)
        in_dim = self.state_dim + self.action_horizon * self.action_dim
        out_dim = self.state_horizon * self.state_dim

        layers: list[nn.Module] = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s_t: torch.Tensor, a_chunk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_t: [B, D_posrot]
            a_chunk: [B, H_a, D_a]  (normalized action head space; no pad columns)
        Returns:
            s_hat: [B, T_s, D_posrot]
        """
        if s_t.ndim != 2 or s_t.shape[-1] != self.state_dim:
            raise ValueError(f"s_t expected [B,{self.state_dim}], got {tuple(s_t.shape)}")
        if a_chunk.ndim != 3:
            raise ValueError(f"a_chunk expected [B,H,D], got {tuple(a_chunk.shape)}")
        if a_chunk.shape[1] != self.action_horizon or a_chunk.shape[2] != self.action_dim:
            raise ValueError(
                f"a_chunk expected [B,{self.action_horizon},{self.action_dim}], "
                f"got {tuple(a_chunk.shape)}"
            )
        x = torch.cat([s_t, a_chunk.reshape(a_chunk.shape[0], -1)], dim=-1)
        out = self.net(x)
        return out.view(-1, self.state_horizon, self.state_dim)


class GroupNormMSE(nn.Module):
    """Group-normalized MSE over pos and rot6d (dataset group std, then sum).

    Maintains running variance per group on supervised frames. Call
    ``update_stats`` on GT batches (no grad) before / during early training,
    or pass ``group_std`` once from a calibration pass.
    """

    def __init__(
        self,
        state_dim: int = POS_ROT_DIM,
        eps: float = 1e-6,
        score_slice: slice = SCORE_FRAME_SLICE,
        aux_slice: slice = slice(9, None),
        score_weight: float = 1.0,
        aux_weight: float = 1.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.eps = float(eps)
        self.score_slice = score_slice
        self.aux_slice = aux_slice
        self.score_weight = float(score_weight)
        self.aux_weight = float(aux_weight)
        pos_idx, rot_idx = pos_rot_slices(state_dim)
        self.register_buffer("pos_idx", pos_idx, persistent=True)
        self.register_buffer("rot_idx", rot_idx, persistent=True)
        # Per-dimension mean/variance on valid frames. Batches accumulate by
        # frame count (not one vote per batch). variance = E[x^2] - E[x]^2.
        self.register_buffer("sum_x", torch.zeros(state_dim), persistent=True)
        self.register_buffer("sum_x2", torch.zeros(state_dim), persistent=True)
        self.register_buffer("count", torch.zeros((), dtype=torch.float64), persistent=True)

    def dim_std(self) -> torch.Tensor:
        """Per-dimension std. Variance is floored at eps before the square root."""
        c = self.count.to(dtype=self.sum_x.dtype).clamp_min(1.0)
        mean = self.sum_x / c
        var = self.sum_x2 / c - mean.square()
        return var.clamp_min(self.eps).sqrt()

    @torch.no_grad()
    def update_stats(self, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """Accumulate per-dim sum and sum-of-squares on supervised frames.

        gt: [B, T, D]. mask: optional [B, T] (True = valid). Frame 0 is excluded
        (same slice as the loss). Each valid frame adds 1 to ``count`` for every dim.
        """
        x = gt[..., SUP_FRAME_SLICE, :].detach()
        if mask is not None:
            m = mask[..., SUP_FRAME_SLICE].to(dtype=x.dtype)
        else:
            m = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        # [B, T, 1] so a temporal mask applies to every dimension.
        m3 = m.unsqueeze(-1)
        self.sum_x.add_((x * m3).sum(dim=(0, 1)).to(self.sum_x.dtype))
        self.sum_x2.add_((x.square() * m3).sum(dim=(0, 1)).to(self.sum_x2.dtype))
        self.count.add_(m.sum().to(dtype=self.count.dtype))

    def _group_mse(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask is not None:
            m = mask.to(dtype=pred.dtype).unsqueeze(-1)
        else:
            m = torch.ones(*pred.shape[:2], 1, device=pred.device, dtype=pred.dtype)
        w = m.sum().clamp_min(1.0)
        std = self.dim_std()

        def _one(idx: torch.Tensor) -> torch.Tensor:
            gstd = std.index_select(0, idx).view(1, 1, -1)
            err = (pred.index_select(-1, idx) - gt.index_select(-1, idx)) / gstd
            return (err.square() * m).sum() / (w * idx.numel())

        pos_l = _one(self.pos_idx)
        rot_l = _one(self.rot_idx)
        return pos_l + rot_l, pos_l, rot_l

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred, gt: [B, T_s, D_posrot]
            mask: optional [B, T_s] temporal validity (True = valid)
        """
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
        score_m = None if mask is None else mask[:, self.score_slice]
        aux_m = None if mask is None else mask[:, self.aux_slice]
        loss_score, pos_s, rot_s = self._group_mse(
            pred[:, self.score_slice], gt[:, self.score_slice], score_m
        )
        if pred.shape[1] > self.aux_slice.start:
            loss_aux, pos_a, rot_a = self._group_mse(
                pred[:, self.aux_slice], gt[:, self.aux_slice], aux_m
            )
        else:
            loss_aux = pred.new_zeros(())
            pos_a = rot_a = loss_aux
        # Also supervise remaining early frames outside score window but before aux
        # (none when score is 1:9 and aux is 9:). Frames in SUP but not score/aux:
        mid = slice(SUP_FRAME_SLICE.start, self.score_slice.start)
        if mid.start < mid.stop:
            mid_m = None if mask is None else mask[:, mid]
            loss_mid, _, _ = self._group_mse(pred[:, mid], gt[:, mid], mid_m)
        else:
            loss_mid = pred.new_zeros(())

        total = (
            self.score_weight * loss_score
            + self.aux_weight * loss_aux
            + self.aux_weight * loss_mid
        )
        return {
            "loss": total,
            "loss_score": loss_score.detach(),
            "loss_aux": loss_aux.detach(),
            "loss_pos_score": pos_s.detach(),
            "loss_rot_score": rot_s.detach(),
            "loss_pos_aux": pos_a.detach(),
            "loss_rot_aux": rot_a.detach(),
            "score_distance": loss_score.detach(),  # D on exec window
        }

    def distance(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample D on the exec window t+1..t+8. Shape [B]. score = -D."""
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
        sl = self.score_slice
        p = pred[:, sl]
        g = gt[:, sl]
        if mask is not None:
            m = mask[:, sl].to(dtype=pred.dtype).unsqueeze(-1)
        else:
            m = torch.ones(*p.shape[:2], 1, device=pred.device, dtype=pred.dtype)
        w = m.sum(dim=(1, 2)).clamp_min(1.0)  # [B]
        std = self.dim_std()

        def _one(idx: torch.Tensor) -> torch.Tensor:
            gstd = std.index_select(0, idx).view(1, 1, -1)
            err = (p.index_select(-1, idx) - g.index_select(-1, idx)) / gstd
            return (err.square() * m).sum(dim=(1, 2)) / (w * idx.numel())

        return _one(self.pos_idx) + _one(self.rot_idx)
