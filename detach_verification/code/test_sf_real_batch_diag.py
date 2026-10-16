"""CPU unit tests with lightweight SF/action modules, NOT full DIAL validation.

Optional STUDENT_MODEL_SOURCE executes the exact supplied SF forward block.
"""
import os
from pathlib import Path
import random
import sys
import textwrap
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sf_real_batch_diag import diagnose, validate_loading, validate_launch


def load_student_block():
    path = os.environ.get("STUDENT_MODEL_SOURCE")
    if not path:
        return None
    source = Path(path).read_text()
    start = source.index("            pred_state_future = None", source.index("def forward("))
    end = source.index("            # SF MSE before matching/CFG", start)
    body = textwrap.indent(textwrap.dedent(source[start:end]), "    ")
    scope = {}
    exec("def student_block(self, raw_backbone_feat, batch_size):\n" + body +
         "    return pred_state_future, proprio_foresight_tokens\n", scope)
    return scope["student_block"]


STUDENT_BLOCK = load_student_block()


class SmallModel(nn.Module):
    def __init__(self, bug=None):
        super().__init__()
        self.backbone = nn.Linear(5, 8)
        self.state_future_projector = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 12))
        self.state_future_to_cond = nn.Linear(4, 8)
        self.proprio_foresight_type_emb = nn.Parameter(torch.zeros(1, 1, 8))
        self.action_head = nn.Linear(8, 2)
        self.use_state_future = self.compute_state_future_loss = self.feed_state_future_to_dit = True
        self.sf_tokens_detach = False
        self.sf_dit_repr = "shat"
        self.loss_combine_mode = "additive"
        self.state_future_loss_weight = 0.05
        self.bridge_type = "golden"
        self.bug = bug

    def _predict_state_future(self, features, batch_size, return_hidden=False):
        return self.state_future_projector(features.mean(1)).reshape(batch_size, 3, 4)

    def _apply_sf_cond_ablations(self, tokens):
        return tokens

    def forward(self, inputs, action_mode=False):
        features = self.backbone(inputs.pop("x")).unsqueeze(1)
        if STUDENT_BLOCK is not None and self.bug is None:
            pred, cond = STUDENT_BLOCK(self, features, features.shape[0])
        else:
            pred = self._predict_state_future(features, features.shape[0])
            should_detach = self.sf_tokens_detach and self.bug != "ignored_switch"
            cond = self.state_future_to_cond(pred.detach() if should_detach else pred)
            cond = cond + self.proprio_foresight_type_emb
        if self.bug == "cut_sf_loss" and self.sf_tokens_detach:
            pred = pred.detach()
        # Includes a direct feature path: detach SF should not remove it.
        action = self.action_head(cond.mean(1) + features.mean(1))
        noise = torch.randn_like(action)
        action_loss = ((action - inputs["y"] - noise * 0.1) ** 2).mean()
        sf_loss = ((pred - inputs["sf"]) ** 2).mean()
        return {"action_loss": action_loss, "state_future_loss": sf_loss,
                "bridge_loss": features.square().mean()}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        self.model = SmallModel()
        self.batch = {"x": torch.randn(2, 5), "y": torch.randn(2, 2), "sf": torch.randn(2, 3, 4)}

    def test_true_routing_and_actual_eval_interventions(self):
        report = diagnose(self.model, self.batch)
        self.assertTrue(report["gradient_routing_checks_passed"])
        self.assertFalse(report["runs"]["detach"]["sf_input_requires_grad"])
        self.assertTrue(report["runs"]["no_detach"]["sf_input_requires_grad"])
        for name in ("zero_projected_sf", "swap_projected_sf"):
            self.assertGreater(report["runs"][name]["cond_delta_max"], 0)
            self.assertNotEqual(report["runs"][name]["action_loss_delta_from_normal"], 0)

    def test_restores_state_rng_and_does_not_update_weights(self):
        self.model.train()
        self.model.backbone.eval()
        self.model.backbone.weight.requires_grad_(False)
        old_modes = [m.training for m in self.model.modules()]
        old_flags = [p.requires_grad for p in self.model.parameters()]
        old_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.action_head.weight.grad = torch.ones_like(self.model.action_head.weight)
        grad = self.model.action_head.weight.grad.clone()
        rng = torch.get_rng_state().clone()
        py_rng = random.getstate()
        np_rng = np.random.get_state()
        diagnose(self.model, self.batch)
        self.assertEqual(old_modes, [m.training for m in self.model.modules()])
        self.assertEqual(old_flags, [p.requires_grad for p in self.model.parameters()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(py_rng, random.getstate())
        np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
        self.assertEqual(np_rng[2:], np.random.get_state()[2:])
        self.assertIn("x", self.batch)
        self.assertTrue(torch.equal(grad, self.model.action_head.weight.grad))
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, old_weights[k]))

    def test_ignored_switch_detected(self):
        self.model.bug = "ignored_switch"
        report = diagnose(self.model, self.batch)
        self.assertFalse(report["gradient_routing_checks_passed"])
        self.assertFalse(report["checks"]["detach_action_disconnected_from_decoder"])

    def test_cut_sf_supervision_detected_and_state_restored(self):
        self.model.bug = "cut_sf_loss"
        with self.assertRaisesRegex(RuntimeError, "differentiable"):
            diagnose(self.model, self.batch)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.sf_tokens_detach)
        self.assertEqual(len(self.model.state_future_to_cond._forward_hooks), 0)

    def test_batch_one_skips_swap(self):
        report = diagnose(self.model, {k: v[:1] for k, v in self.batch.items()})
        self.assertIn("skipped", report["runs"]["swap_projected_sf"])

    def test_frozen_decoder_is_not_silently_unfrozen(self):
        self.model.state_future_projector.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            diagnose(self.model, self.batch)

    def test_missing_loss_detected(self):
        self.model.compute_state_future_loss = False
        with self.assertRaisesRegex(RuntimeError, "Required flag"):
            diagnose(self.model, self.batch)

    def test_loading_info_strict(self):
        validate_loading({"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []})
        for info in (None, {"missing_keys": ["state_future_projector.0.weight"]},
                     {"unexpected_keys": ["old_decoder.weight"]}, {"mismatched_keys": ["shape"]}):
            with self.assertRaises(RuntimeError):
                validate_loading(info)

    def test_launch_rejects_overwrite_partial_load_and_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            cfg = SimpleNamespace(num_gpus=1, resume=False, resume_pretrained_option="all",
                                  resume_vlm_path=None, resume_action_head_path=None,
                                  base_model_path=str(checkpoint), output_dir=str(root / "output"), lora_rank=0)
            with self.assertRaisesRegex(RuntimeError, "weight files"):
                validate_launch(cfg)
            # Path-validation fixture only; never passed to a weight loader.
            (checkpoint / "model.safetensors").touch()
            validate_launch(cfg)
            cfg.resume_pretrained_option = "action_head"
            with self.assertRaisesRegex(RuntimeError, "complete checkpoint"):
                validate_launch(cfg)
            cfg.resume_pretrained_option = "all"
            (root / "output" / "experiment_cfg").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "NEW diagnostic"):
                validate_launch(cfg)


if __name__ == "__main__":
    print("Testing student's extracted SF forward block:", STUDENT_BLOCK is not None)
    unittest.main()
