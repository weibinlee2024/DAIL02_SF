"""Group-relative policy loss for the Gaussian intent policy."""

from __future__ import annotations

import torch


def group_advantage(reward: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize reward over the group axis. reward is [B, G]."""
    if reward.ndim != 2 or reward.shape[1] < 2:
        raise ValueError(f"reward expected [B, G>=2], got {tuple(reward.shape)}")
    mean = reward.mean(dim=1, keepdim=True)
    std = reward.std(dim=1, keepdim=True, unbiased=False)
    advantage = (reward - mean) / std.clamp_min(eps)
    return torch.where(std < eps, torch.zeros_like(advantage), advantage)


def grpo_policy_loss(
    log_prob: torch.Tensor,
    log_prob_ref: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clipped importance ratio. Advantage is stop-grad. Returns loss and ratio."""
    if clip_eps <= 0 or clip_eps >= 1:
        raise ValueError(f"clip_eps must lie in (0, 1), got {clip_eps}")
    log_ratio = (log_prob - log_prob_ref).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    adv = advantage.detach()
    surrogate = torch.min(ratio * adv, clipped * adv)
    return -surrogate.mean(), ratio.detach()


def gaussian_kl(mu: torch.Tensor, mu_ref: torch.Tensor, sigma: float) -> torch.Tensor:
    """Analytic KL for two diagonal Gaussians that share sigma, averaged over elements."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if mu.shape != mu_ref.shape:
        raise ValueError(f"mu {tuple(mu.shape)} != ref {tuple(mu_ref.shape)}")
    return (mu - mu_ref).square().mean() / (2.0 * float(sigma) ** 2)
