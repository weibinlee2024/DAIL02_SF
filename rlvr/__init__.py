"""Constraint-space RLVR post-training for DIAL System-2.

The supervised forward and the online WM Best-of-N selector are unchanged.
This package samples a group of intents, decodes them with a frozen System-1,
and scores the frozen world model's wrist rollout.
"""

from gr00t.rlvr.grpo import gaussian_kl, group_advantage, grpo_policy_loss
from gr00t.rlvr.intent_policy import expand_gaussian, gaussian_log_prob
from gr00t.rlvr.rewards import ConstraintReward, constraint_reward

__all__ = [
    "ConstraintReward",
    "constraint_reward",
    "expand_gaussian",
    "gaussian_kl",
    "gaussian_log_prob",
    "group_advantage",
    "grpo_policy_loss",
]
