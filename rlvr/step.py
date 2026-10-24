"""One CS-RLVR update: group intents, frozen decode, frozen world-model reward."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from gr00t.rlvr.grpo import gaussian_kl, group_advantage, grpo_policy_loss
from gr00t.rlvr.intent_policy import expand_gaussian, gaussian_log_prob
from gr00t.rlvr.rewards import constraint_reward
from gr00t.rlvr.system2 import (
    align_visual_target,
    decode_action_group,
    intent_means,
    prepare_rlvr_context,
    sf_tokens_from_samples,
    use_modules,
)
from gr00t.rlvr.wm_loader import FrozenWorldModel


def _shared_action_noise(model, batch: int, group: int) -> torch.Tensor:
    head = model.action_head
    noise = torch.randn(
        batch,
        head.config.action_horizon,
        head.config.action_dim,
        device=head.device,
        dtype=head.dtype,
    )
    return noise.repeat_interleave(group, dim=0)


def _add_gaussian(
    log_prob: torch.Tensor,
    log_prob_ref: torch.Tensor,
    kl: torch.Tensor,
    mu: torch.Tensor,
    mu_ref: torch.Tensor,
    sigma: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = expand_gaussian(mu, group_size, sigma)
    log_prob = log_prob + gaussian_log_prob(samples.float(), mu.float(), sigma)
    log_prob_ref = log_prob_ref + gaussian_log_prob(samples.float(), mu_ref.float(), sigma)
    kl = kl + gaussian_kl(mu.float(), mu_ref.float(), sigma)
    return log_prob, log_prob_ref, kl, samples


def cs_rlvr_loss(
    model,
    inputs: dict,
    world_model: FrozenWorldModel,
    ref_modules: dict[str, torch.nn.Module],
    *,
    group_size: int,
    sigma_visual: float,
    sigma_sf: float,
    lambda_obs: float,
    lambda_phy: float,
    lambda_hard: float,
    alpha: float,
    beta: float,
    clip_eps: float,
    acc_limit: float,
    workspace_bound: float,
    hand_bound: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the scalar loss and a detached log dict."""
    context = prepare_rlvr_context(model, inputs)
    means = intent_means(model, context["raw"])
    with torch.no_grad(), use_modules(model, **ref_modules):
        reference = intent_means(model, context["raw"])

    batch = means["mu_visual"].shape[0]
    log_prob = torch.zeros(batch, group_size, device=means["mu_visual"].device, dtype=torch.float32)
    log_prob_ref = torch.zeros(batch, group_size, device=means["mu_visual"].device, dtype=torch.float32)
    kl = torch.zeros((), device=means["mu_visual"].device, dtype=torch.float32)
    log_prob, log_prob_ref, kl, visual_samples = _add_gaussian(
        log_prob,
        log_prob_ref,
        kl,
        means["mu_visual"],
        reference["mu_visual"],
        sigma_visual,
        group_size,
    )

    state_samples = None
    token_samples = None
    if means["sf_space"] == "state":
        log_prob, log_prob_ref, kl, state_samples = _add_gaussian(
            log_prob,
            log_prob_ref,
            kl,
            means["mu_sf_state"],
            reference["mu_sf_state"],
            sigma_sf,
            group_size,
        )
        if model.feed_state_future_to_dit:
            token_samples = sf_tokens_from_samples(model, state_samples, "state")
    elif means["sf_space"] == "token":
        log_prob, log_prob_ref, kl, token_samples = _add_gaussian(
            log_prob,
            log_prob_ref,
            kl,
            means["mu_sf_token"],
            reference["mu_sf_token"],
            sigma_sf,
            group_size,
        )
        state_samples = (
            means["mu_sf_state"]
            .detach()
            .unsqueeze(1)
            .expand(batch, group_size, *means["mu_sf_state"].shape[1:])
            .contiguous()
        )

    noise = _shared_action_noise(model, batch, group_size)
    with torch.no_grad():
        actions = decode_action_group(
            model,
            visual_samples,
            context["obs"],
            token_samples,
            context["action_inputs"],
            noise,
        )
    with torch.no_grad(), torch.autocast(device_type=actions.device.type, enabled=False):
        wrist = world_model.rollout(
            context["s_t"].repeat_interleave(group_size, dim=0),
            actions,
        )
    target = align_visual_target(means["mu_visual"].detach(), context["goal"])
    reward = constraint_reward(
        wrist,
        visual_samples,
        target,
        state_samples,
        lambda_obs=lambda_obs,
        lambda_phy=lambda_phy,
        lambda_hard=lambda_hard,
        acc_limit=acc_limit,
        workspace_bound=workspace_bound,
        hand_bound=hand_bound,
    )
    advantage = group_advantage(reward.total)
    policy_loss, ratio = grpo_policy_loss(log_prob, log_prob_ref, advantage, clip_eps=clip_eps)
    if target is None:
        anchor = means["mu_visual"].new_zeros(())
    else:
        anchor = F.mse_loss(means["mu_visual"].float(), target.float())
    loss = policy_loss + float(beta) * kl + float(alpha) * anchor
    stats = {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "kl": float(kl.detach()),
        "anchor": float(anchor.detach()),
        "reward": float(reward.total.mean()),
        "r_obs": float(reward.r_obs.mean()),
        "r_phy": float(reward.r_phy.mean()),
        "r_hard": float(reward.r_hard.mean()),
        "accel_penalty": float(reward.accel_penalty.mean()),
        "workspace_penalty": float(reward.workspace_penalty.mean()),
        "hand_penalty": float(reward.hand_penalty.mean()),
        "ratio": float(ratio.mean()),
        "advantage_abs": float(advantage.detach().abs().mean()),
        "visual_target": target is not None,
    }
    return loss, stats
