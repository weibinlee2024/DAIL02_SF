"""CPU checks for the CS-RLVR reward and the group-relative loss."""

from __future__ import annotations

import unittest

import torch

from gr00t.rlvr.grpo import gaussian_kl, group_advantage, grpo_policy_loss
from gr00t.rlvr.intent_policy import expand_gaussian, gaussian_log_prob
from gr00t.rlvr.rewards import acceleration_penalty, constraint_reward, hand_penalty, workspace_penalty


class RewardTests(unittest.TestCase):
    def test_acceleration_penalizes_a_jump(self):
        smooth = torch.zeros(2, 17, 18)
        smooth[:, :, 0] = torch.linspace(0, 0.2, 17)
        jumpy = smooth.clone()
        jumpy[:, 4, 0] += 5.0
        self.assertGreater(
            acceleration_penalty(jumpy, acc_limit=0.5).mean().item(),
            acceleration_penalty(smooth, acc_limit=0.5).mean().item(),
        )

    def test_workspace_and_hand_bounds(self):
        inside = torch.zeros(1, 17, 18)
        outside = inside.clone()
        outside[:, 3, 0] = 4.0
        self.assertEqual(workspace_penalty(inside, bound=3.0).item(), 0.0)
        self.assertGreater(workspace_penalty(outside, bound=3.0).item(), 0.0)

        state = torch.zeros(1, 17, 30)
        self.assertEqual(hand_penalty(state, bound=1.5).item(), 0.0)
        state[:, 2, 20] = 3.0
        self.assertGreater(hand_penalty(state, bound=1.5).item(), 0.0)
        wrist_only = torch.zeros(1, 17, 18)
        self.assertEqual(hand_penalty(wrist_only, bound=1.5).item(), 0.0)

    def test_group_reward_shape_is_detached(self):
        wrist = torch.zeros(4, 17, 18)
        visual = torch.zeros(2, 2, 3, 4)
        target = torch.zeros(2, 3, 4)
        state = torch.zeros(2, 2, 17, 30)
        reward = constraint_reward(
            wrist,
            visual,
            target,
            state,
            lambda_obs=0.2,
            lambda_phy=1.0,
            lambda_hard=1.0,
            acc_limit=1.0,
            workspace_bound=3.0,
            hand_bound=1.5,
        )
        self.assertEqual(tuple(reward.total.shape), (2, 2))
        self.assertFalse(reward.total.requires_grad)


class PolicyTests(unittest.TestCase):
    def test_equal_rewards_have_zero_advantage(self):
        advantage = group_advantage(torch.ones(3, 4))
        self.assertTrue(torch.all(advantage == 0))

    def test_positive_advantage_pulls_the_mean_toward_the_sample(self):
        mu = torch.nn.Parameter(torch.zeros(2, 3))
        samples = torch.ones(2, 2, 3)
        loss = -gaussian_log_prob(samples, mu, 0.5).mean()
        loss.backward()
        self.assertIsNotNone(mu.grad)
        self.assertTrue(torch.all(mu.grad < 0))

    def test_samples_do_not_carry_gradient(self):
        mu = torch.nn.Parameter(torch.zeros(2, 4))
        samples = expand_gaussian(mu, group_size=3, sigma=0.1)
        self.assertFalse(samples.requires_grad)
        self.assertEqual(tuple(samples.shape), (2, 3, 4))

    def test_clipped_ratio_is_finite(self):
        log_prob = torch.zeros(2, 4, requires_grad=True)
        log_prob_ref = torch.zeros(2, 4)
        advantage = torch.tensor([[1.0, -1.0, 0.5, -0.5], [0.2, -0.2, 0.0, 1.0]])
        loss, ratio = grpo_policy_loss(log_prob, log_prob_ref, advantage, clip_eps=0.2)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(ratio <= 1.2 + 1e-5))
        loss.backward()
        self.assertIsNotNone(log_prob.grad)

    def test_kl_is_zero_at_the_reference(self):
        mu = torch.randn(2, 3, 5)
        self.assertAlmostEqual(float(gaussian_kl(mu, mu, sigma=0.2)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
