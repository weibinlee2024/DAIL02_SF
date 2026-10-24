"""Load the frozen action-conditioned world model used as the verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gr00t.model.wm_verifier.action_conditioned_dynamics import (
    POS_ROT_DIM,
    ActionConditionedDynamics,
)


@dataclass
class FrozenWorldModel:
    model: ActionConditionedDynamics
    action_dim: int
    action_horizon: int
    state_horizon: int

    def rollout(self, s_t: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """s_t is [N, 18], action_chunk is [N, H, D] in the world model's space."""
        action = action_chunk[:, : self.action_horizon, : self.action_dim].float()
        state = s_t[:, :POS_ROT_DIM].float()
        return self.model(state, action)


def load_frozen_world_model(path: str, device: torch.device | str) -> FrozenWorldModel:
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu")
    meta = ckpt.get("meta") or {}
    saved_args = ckpt.get("args") or {}
    if "action_dim" not in meta or "action_horizon" not in meta or "state_horizon" not in meta:
        raise RuntimeError(f"world model checkpoint {path} is missing action/state meta")
    model = ActionConditionedDynamics(
        state_dim=POS_ROT_DIM,
        action_dim=int(meta["action_dim"]),
        action_horizon=int(meta["action_horizon"]),
        state_horizon=int(meta["state_horizon"]),
        hidden_dim=int(saved_args.get("hidden_dim", 512)),
        num_layers=int(saved_args.get("num_layers", 3)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return FrozenWorldModel(
        model=model,
        action_dim=int(meta["action_dim"]),
        action_horizon=int(meta["action_horizon"]),
        state_horizon=int(meta["state_horizon"]),
    )
