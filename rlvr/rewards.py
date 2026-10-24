"""Constraint rewards on a frozen world-model wrist rollout.

The score uses the rollout itself: excessive wrist acceleration, wrist
workspace bounds, and hand-joint bounds on the sampled proprio forecast.
Visual alignment is a separate, smaller term against the frozen future
image embedding. The online Best-of-N distance between the world model
and the policy's own forecast is not this reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Frames t+1..t+8, the same window the frozen verifier scores.
SCORE_FRAMES = slice(1, 9)

# Wrist xyz inside the 18-d pos+rot6d layout:
# [r_pos 0:3 | r_rot 3:9 | l_pos 9:12 | l_rot 12:18]
WRIST_XYZ = (0, 1, 2, 9, 10, 11)
# state_future layout after the 18-d wrist block: right hand (6) then left hand (6).
HAND_SLICE = slice(18, 30)


@dataclass
class ConstraintReward:
    """Per-candidate terms. Each tensor is [B, G] and already stop-grad ready."""

    total: torch.Tensor
    r_obs: torch.Tensor
    r_phy: torch.Tensor
    r_hard: torch.Tensor
    accel_penalty: torch.Tensor
    workspace_penalty: torch.Tensor
    hand_penalty: torch.Tensor


def _reduce_mean(values: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, values.ndim))
    if not reduce_dims:
        return values
    return values.mean(dim=reduce_dims)


def acceleration_penalty(wrist_rollout: torch.Tensor, acc_limit: float) -> torch.Tensor:
    """Mean ReLU(|acceleration| - limit) of both wrists. wrist_rollout is [N, T, 18]."""
    if wrist_rollout.ndim != 3 or wrist_rollout.shape[-1] < 18:
        raise ValueError(f"wrist rollout expected [N, T, >=18], got {tuple(wrist_rollout.shape)}")
    window = wrist_rollout[:, SCORE_FRAMES]
    pos = window[..., WRIST_XYZ]
    if pos.shape[1] < 3:
        return wrist_rollout.new_zeros(wrist_rollout.shape[0])
    velocity = pos[:, 1:] - pos[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    excess = (acceleration.abs() - float(acc_limit)).clamp(min=0)
    return _reduce_mean(excess)


def workspace_penalty(wrist_rollout: torch.Tensor, bound: float) -> torch.Tensor:
    """Fraction of wrist-xyz values outside ±bound on the scored frames."""
    window = wrist_rollout[:, SCORE_FRAMES]
    pos = window[..., WRIST_XYZ]
    if pos.shape[1] == 0:
        return wrist_rollout.new_zeros(wrist_rollout.shape[0])
    return _reduce_mean((pos.abs() > float(bound)).to(pos.dtype))


def hand_penalty(state_future: torch.Tensor | None, bound: float) -> torch.Tensor:
    """Fraction of sampled hand joints outside ±bound. Zero when hands are absent."""
    if state_future is None:
        raise ValueError("state_future is required")
    if state_future.shape[-1] < HAND_SLICE.stop:
        return state_future.new_zeros(state_future.shape[0])
    hands = state_future[:, 1:, HAND_SLICE]
    if hands.shape[1] == 0:
        return state_future.new_zeros(state_future.shape[0])
    return _reduce_mean((hands.abs() > float(bound)).to(hands.dtype))


def visual_mse(samples: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor | None:
    """Per-candidate MSE. samples is [B, G, ...], target is [B, ...]."""
    if target is None:
        return None
    target_g = target.unsqueeze(1)
    if samples.shape[0] != target_g.shape[0] or samples.shape[2:] != target_g.shape[2:]:
        return None
    err = (samples - target_g).square()
    reduce_dims = tuple(range(2, err.ndim))
    return err.mean(dim=reduce_dims)


def constraint_reward(
    wrist_rollout: torch.Tensor,
    visual_samples: torch.Tensor,
    visual_target: torch.Tensor | None,
    state_samples: torch.Tensor | None,
    *,
    lambda_obs: float,
    lambda_phy: float,
    lambda_hard: float,
    acc_limit: float,
    workspace_bound: float,
    hand_bound: float,
) -> ConstraintReward:
    """Build R = λ_obs r_obs + λ_phy r_phy + λ_hard r_hard, shaped [B, G]."""
    batch, group = visual_samples.shape[:2]
    flat = batch * group

    def _group(values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] != flat:
            raise ValueError(f"expected {flat} candidates, got {values.shape[0]}")
        return values.view(batch, group)

    accel = acceleration_penalty(wrist_rollout, acc_limit)
    workspace = workspace_penalty(wrist_rollout, workspace_bound)
    if state_samples is None:
        hands = wrist_rollout.new_zeros(flat)
    else:
        hands = hand_penalty(state_samples.reshape(flat, *state_samples.shape[2:]), hand_bound)
    r_phy = -_group(accel)
    r_hard = -_group(workspace + hands)
    obs = visual_mse(visual_samples, visual_target)
    if obs is None:
        r_obs = visual_samples.new_zeros(batch, group)
    else:
        r_obs = -obs
    total = (
        float(lambda_obs) * r_obs
        + float(lambda_phy) * r_phy
        + float(lambda_hard) * r_hard
    )
    return ConstraintReward(
        total=total.detach(),
        r_obs=r_obs.detach(),
        r_phy=r_phy.detach(),
        r_hard=r_hard.detach(),
        accel_penalty=_group(accel).detach(),
        workspace_penalty=_group(workspace).detach(),
        hand_penalty=_group(hands).detach(),
    )
