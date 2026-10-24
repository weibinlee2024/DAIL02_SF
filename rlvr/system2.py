"""System-2 intent packing for CS-RLVR.

Runs the frozen VLM once, then reads the visual projector and the proprio
decoder. The DiT condition is always the predicted intent. Future image
embeddings are kept only as the visual reward target.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.wm_verifier.action_conditioned_dynamics import POS_ROT_DIM


def trainable_module_names(model) -> list[str]:
    names = ["bridge_projector"]
    if hasattr(model, "state_future_decoder"):
        names.append("state_future_decoder")
    elif hasattr(model, "state_future_projector"):
        names.append("state_future_projector")
    missing = [name for name in names if not hasattr(model, name)]
    if missing:
        raise RuntimeError(f"CS-RLVR requires modules {missing}")
    return names


@contextmanager
def use_modules(model, **modules) -> Iterator[None]:
    saved = {}
    for name, module in modules.items():
        if module is None or not hasattr(model, name):
            continue
        saved[name] = getattr(model, name)
        setattr(model, name, module)
    try:
        yield
    finally:
        for name, module in saved.items():
            setattr(model, name, module)


def current_wrist(state: torch.Tensor) -> torch.Tensor:
    if state.ndim == 3:
        frame = state[:, -1]
    elif state.ndim == 2:
        frame = state
    else:
        raise ValueError(f"state expected [B, D] or [B, T, D], got {tuple(state.shape)}")
    if frame.shape[-1] < POS_ROT_DIM:
        raise ValueError(f"state width {frame.shape[-1]} is below wrist pos+rot6d ({POS_ROT_DIM})")
    return frame[:, :POS_ROT_DIM]


def repeat_feature(feature: BatchFeature, times: int) -> BatchFeature:
    data = {}
    for key, value in feature.items():
        if torch.is_tensor(value):
            data[key] = value.repeat_interleave(times, dim=0)
        else:
            data[key] = value
    return BatchFeature(data=data)


def flatten_group(samples: torch.Tensor) -> torch.Tensor:
    batch, group = samples.shape[:2]
    return samples.reshape(batch * group, *samples.shape[2:])


def _observation_embeds(model, backbone_inputs, batch_size: int) -> tuple[torch.Tensor, int]:
    if model.use_dino_vision:
        dino_img = backbone_inputs["dino_obs_images"]
        flat = dino_img.view(-1, *dino_img.shape[2:])
        raw = model._extract_dino(flat, downsample_to_n=None)
        num_views = raw.shape[0] // batch_size
        obs = raw.reshape(batch_size, num_views * 256, -1)
        tokens_per_image = 256
    else:
        if "act_obs_image_pixel_values" in backbone_inputs:
            prefix = "act_obs_image_"
            obs_input = {
                key.removeprefix(prefix): value
                for key, value in backbone_inputs.items()
                if key.startswith(prefix)
            }
        else:
            obs_input = {
                key.removeprefix("vlm_"): value
                for key, value in backbone_inputs.items()
                if key.startswith("vlm_")
            }
            obs_input.pop("image_sizes", None)
            obs_input.pop("input_ids", None)
            obs_input.pop("attention_mask", None)
        obs = model.bridge_vision_model(*obs_input.values())
        if model.action_only_one_obs:
            obs = obs.reshape(batch_size, -1, model.num_bridge_tokens, obs.shape[-1])[:, -1]
        else:
            obs = obs.reshape(batch_size, -1, obs.shape[-1])
        tokens_per_image = model.num_bridge_tokens
        num_views = obs.shape[1] // tokens_per_image
    if model.bridge_loss_type == "cosine":
        obs = F.normalize(obs, p=2, dim=-1)
    if model.use_image_type_embedding:
        for view in range(num_views):
            start = view * tokens_per_image
            end = (view + 1) * tokens_per_image
            obs[:, start:end] = obs[:, start:end] + model.image_type_embedding.weight[view + 1]
    return obs, num_views


def _goal_target(model, backbone_inputs, obs_embeds: torch.Tensor, batch_size: int) -> torch.Tensor | None:
    goal = None
    if model.goal_image_type == "current":
        goal = obs_embeds
    elif model.goal_image_type == "future":
        has_dino_goal = "dino_goal_images" in backbone_inputs
        has_goal_pixels = any(key.startswith("goal_image_") for key in backbone_inputs)
        if model.use_dino_goal and has_dino_goal:
            dino_goal = backbone_inputs["dino_goal_images"]
            flat = dino_goal.view(-1, *dino_goal.shape[2:])
            raw = model._extract_dino(flat, downsample_to_n=model.num_bridge_tokens)
            views = raw.shape[0] // batch_size
            goal = raw.view(batch_size, views * model.num_bridge_tokens, -1)
        elif (not model.use_dino_goal) and has_goal_pixels:
            prefix = "goal_image_"
            goal_input = {
                key.removeprefix(prefix): value
                for key, value in backbone_inputs.items()
                if key.startswith(prefix)
            }
            goal = model._run_bridge_vision(model.bridge_goal_model, goal_input, batch_size)
    else:
        raise NotImplementedError(f"goal_image_type={model.goal_image_type!r}")
    if goal is not None and model.bridge_loss_type == "cosine":
        goal = F.normalize(goal, p=2, dim=-1)
    return goal


def align_visual_target(mu_visual: torch.Tensor, goal: torch.Tensor | None) -> torch.Tensor | None:
    """Match the future embedding to the predicted bridge tokens, averaging views if needed."""
    if goal is None:
        return None
    if goal.shape == mu_visual.shape:
        return goal
    if (
        goal.ndim == 3
        and mu_visual.ndim == 3
        and goal.shape[0] == mu_visual.shape[0]
        and goal.shape[-1] == mu_visual.shape[-1]
        and mu_visual.shape[1] > 0
        and goal.shape[1] % mu_visual.shape[1] == 0
    ):
        views = goal.shape[1] // mu_visual.shape[1]
        return goal.view(goal.shape[0], views, mu_visual.shape[1], goal.shape[-1]).mean(dim=1)
    return None


def prepare_rlvr_context(model, inputs: dict) -> dict:
    """Frozen observation context. The returned backbone features are detached."""
    if not model.use_bridge:
        raise RuntimeError("CS-RLVR requires use_bridge=True")
    backbone_inputs, action_inputs = model.prepare_input(inputs)
    with torch.no_grad():
        backbone_outputs = model.backbone(backbone_inputs)
        sliced = {key: value[:, -model.num_bridge_tokens :] for key, value in backbone_outputs.items()}
        raw = sliced["backbone_features"].detach()
        batch_size = raw.shape[0]
        obs, _num_views = _observation_embeds(model, backbone_inputs, batch_size)
        goal = _goal_target(model, backbone_inputs, obs, batch_size)
    state = action_inputs["state"] if "state" in action_inputs else action_inputs.state
    return {
        "raw": raw,
        "obs": obs.detach(),
        "goal": None if goal is None else goal.detach(),
        "action_inputs": action_inputs,
        "s_t": current_wrist(state).detach(),
    }


def intent_means(model, raw: torch.Tensor) -> dict:
    """Visual token mean and proprio mean. Gradients stay in the active modules.

    ``sf_space`` is ``state`` when the sampled object is Ŝ (shat / no DiT feed),
    and ``token`` when the sampled object is the mid-hidden DiT condition.
    """
    batch_size = raw.shape[0]
    mu_visual = model.bridge_projector(raw)
    if model.bridge_loss_type == "cosine":
        mu_visual = F.normalize(mu_visual, p=2, dim=-1)
    mu_sf_state = None
    mu_sf_token = None
    sf_space = "none"
    if model.use_state_future:
        feed = bool(model.feed_state_future_to_dit)
        if feed and getattr(model, "sf_dit_repr", "shat") == "mid_hidden":
            mu_sf_state, mu_sf_token = model._predict_state_future(
                raw, batch_size, return_hidden=True
            )
            sf_space = "token"
        else:
            mu_sf_state = model._predict_state_future(raw, batch_size, return_hidden=False)
            sf_space = "state"
    return {
        "mu_visual": mu_visual,
        "mu_sf_state": mu_sf_state,
        "mu_sf_token": mu_sf_token,
        "sf_space": sf_space,
    }


def sf_tokens_from_samples(model, sf_samples: torch.Tensor | None, sf_space: str) -> torch.Tensor | None:
    """Map grouped proprio samples to DiT tokens. Samples stay detached."""
    if sf_samples is None or sf_space == "none":
        return None
    flat = flatten_group(sf_samples)
    if sf_space == "state":
        if not hasattr(model, "state_future_to_cond"):
            raise RuntimeError("shat SF samples require state_future_to_cond")
        with torch.no_grad():
            tokens = model.state_future_to_cond(flat)
    elif sf_space == "token":
        tokens = flat
    else:
        raise ValueError(f"unknown sf_space={sf_space!r}")
    batch, group = sf_samples.shape[:2]
    return tokens.view(batch, group, *tokens.shape[1:])


def pack_intent(model, visual: torch.Tensor, obs: torch.Tensor, sf_tokens: torch.Tensor | None) -> BatchFeature:
    """visual/obs are [N, L, H]. Type embeddings match the supervised DiT pack."""
    if model.use_image_type_embedding and not model.omit_image_type_embedding_for_goal:
        visual = visual + model.image_type_embedding.weight[0].to(dtype=visual.dtype, device=visual.device)
    parts = [visual, obs]
    if sf_tokens is not None:
        if not hasattr(model, "proprio_foresight_type_emb"):
            raise RuntimeError("proprio foresight tokens require proprio_foresight_type_emb")
        sf_tokens = sf_tokens + model.proprio_foresight_type_emb.to(
            dtype=sf_tokens.dtype, device=sf_tokens.device
        )
        parts.append(sf_tokens)
    bridge = torch.cat(parts, dim=1)
    mask = torch.ones(bridge.shape[:-1], device=bridge.device, dtype=bridge.dtype)
    return BatchFeature(data={"backbone_features": bridge, "backbone_attention_mask": mask})


def decode_action_group(
    model,
    visual_samples: torch.Tensor,
    obs: torch.Tensor,
    sf_tokens: torch.Tensor | None,
    action_inputs: BatchFeature,
    action_noise: torch.Tensor,
) -> torch.Tensor:
    """Frozen System-1 decode. visual_samples is [B, G, L, H]."""
    group = visual_samples.shape[1]
    visual = flatten_group(visual_samples)
    obs_g = obs.repeat_interleave(group, dim=0)
    sf = None if sf_tokens is None else flatten_group(sf_tokens)
    packed = pack_intent(model, visual, obs_g, sf)
    repeated_inputs = repeat_feature(action_inputs, group)
    prediction = model.action_head.get_action(
        packed,
        repeated_inputs,
        guidance_scale=1.0,
        action_noise=action_noise,
    )
    return prediction["action_pred"]
