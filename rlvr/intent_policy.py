"""Gaussian intent policy around a deterministic System-2 mean.

The sampled intent is treated as a constant. The log-density is evaluated at
that constant, so the gradient reaches the mean and not the reparameterization
noise. A mean reduction keeps the importance ratio from scaling with token count.
"""

from __future__ import annotations

import math

import torch


def expand_gaussian(mu: torch.Tensor, group_size: int, sigma: float) -> torch.Tensor:
    """Draw group_size samples around mu. Returns a detached tensor [B, G, ...]."""
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    sample_shape = (mu.shape[0], group_size, *mu.shape[1:])
    noise = torch.randn(sample_shape, device=mu.device, dtype=mu.dtype)
    samples = mu.unsqueeze(1) + float(sigma) * noise
    return samples.detach()


def gaussian_log_prob(samples: torch.Tensor, mu: torch.Tensor, sigma: float) -> torch.Tensor:
    """Mean per-element Gaussian log-density. samples is [B, G, ...], mu is [B, ...]."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    mu_g = mu.unsqueeze(1)
    if samples.shape[0] != mu_g.shape[0] or samples.shape[2:] != mu_g.shape[2:]:
        raise ValueError(
            f"samples {tuple(samples.shape)} are incompatible with mu {tuple(mu.shape)}"
        )
    var = float(sigma) ** 2
    centered = samples - mu_g
    log_prob = -0.5 * ((centered * centered) / var + math.log(2.0 * math.pi * var))
    reduce_dims = tuple(range(2, log_prob.ndim))
    return log_prob.mean(dim=reduce_dims)
