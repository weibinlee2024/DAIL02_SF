"""Online Best-of-N selection with the frozen WM verifier.

Matches offline scoring: score = -GroupNormMSE(ŝ, pred_sf) on frames t+1..t+8.
``s_t`` is the current wrist pos+rot6d in the policy's state normalization space.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gr00t.model.wm_verifier.action_conditioned_dynamics import (
    POS_ROT_DIM,
    ActionConditionedDynamics,
    GroupNormMSE,
)

# Same first-18 layout as WMPackTransform / offline verifier.
_ST_KEYS: Tuple[str, ...] = (
    "state.wrist_r_pos",
    "state.wrist_r_rot6d",
    "state.wrist_l_pos",
    "state.wrist_l_rot6d",
)


def _as_last_frame(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[-1]
    if arr.ndim == 3:
        return arr[0, -1]
    raise ValueError(f"unexpected state shape {arr.shape}")


def _normalize(x: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    mean_a = np.asarray(mean, dtype=np.float32)
    std_a = np.asarray(std, dtype=np.float32)
    std_a = np.maximum(std_a, 1e-6)
    return (x - mean_a) / std_a


class OnlineWMBoN:
    """Load WM + criterion; extract s_t; pick argmax score among K action chunks."""

    def __init__(
        self,
        wm_ckpt: str,
        policy_metadata: Any,
        device: torch.device | str = "cuda",
        bon_k: int = 8,
    ):
        self.device = torch.device(device)
        self.bon_k = max(int(bon_k), 1)
        ckpt = torch.load(wm_ckpt, map_location="cpu")
        meta = ckpt.get("meta") or {}
        args_saved = ckpt.get("args") or {}
        self.action_dim = int(meta["action_dim"])
        self.action_horizon = int(meta["action_horizon"])
        self.state_horizon = int(meta["state_horizon"])
        self.norm_fingerprint = meta.get("norm_fingerprint")
        model = ActionConditionedDynamics(
            state_dim=POS_ROT_DIM,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            state_horizon=self.state_horizon,
            hidden_dim=int(args_saved.get("hidden_dim", 512)),
            num_layers=int(args_saved.get("num_layers", 3)),
        )
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        crit_sd = ckpt.get("crit") or {}
        if "sum_x" not in crit_sd:
            raise RuntimeError("WM checkpoint missing variance stats (sum_x); retrain first.")
        crit = GroupNormMSE(state_dim=POS_ROT_DIM)
        crit.load_state_dict(crit_sd)
        if float(crit.count.item()) <= 0:
            raise RuntimeError("WM checkpoint has count=0 variance.")
        crit.to(self.device).eval()
        self.wm = model
        self.crit = crit

        stats = getattr(policy_metadata, "statistics", None)
        if stats is None and isinstance(policy_metadata, dict):
            stats = policy_metadata.get("statistics")
        if hasattr(stats, "state"):
            state_stats = stats.state
        elif isinstance(stats, dict):
            state_stats = stats.get("state") or stats
        else:
            raise RuntimeError("policy metadata has no statistics.state")
        self._means: Dict[str, List[float]] = {}
        self._stds: Dict[str, List[float]] = {}
        for key in _ST_KEYS:
            short = key.split(".", 1)[-1]
            st = None
            if hasattr(state_stats, short):
                st = getattr(state_stats, short)
            elif isinstance(state_stats, dict):
                st = state_stats.get(key) or state_stats.get(short)
            if st is None:
                raise KeyError(f"missing stats for {key}")
            mean = st.mean if hasattr(st, "mean") else st["mean"]
            std = st.std if hasattr(st, "std") else st["std"]
            self._means[key] = [float(x) for x in mean]
            self._stds[key] = [float(x) for x in std]
        self._call_idx = 0
        print(
            f"[WM-BoN] loaded k={self.bon_k} action_dim={self.action_dim} "
            f"H={self.action_horizon} fingerprint={self.norm_fingerprint}",
            flush=True,
        )

    def extract_s_t_from_normalized(self, normalized_input: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Read wrist pos+rot6d from post-ConcatTransform ``state`` (first 18 dims).

        Matches WM training / offline packing: same mean_std space as policy state.
        Returns None if ``state`` is missing or too narrow.
        """
        state = normalized_input.get("state")
        if state is None:
            return None
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if not isinstance(state, torch.Tensor):
            return None
        # Expected shapes: [T, D], [B, T, D], or [D]
        if state.ndim == 1:
            flat = state
        elif state.ndim == 2:
            flat = state[-1]
        elif state.ndim == 3:
            flat = state[0, -1]
        else:
            return None
        if flat.shape[-1] < POS_ROT_DIM:
            return None
        return flat[:POS_ROT_DIM].detach().float().to(self.device)

    def extract_s_t(self, observations: Dict[str, Any]) -> torch.Tensor:
        """Fallback: normalize raw wrist keys (pos only is mean_std in data config)."""
        parts = []
        for key in _ST_KEYS:
            if key not in observations:
                raise KeyError(f"obs missing {key} (needed for WM s_t)")
            raw = _as_last_frame(observations[key])
            # gausNorm data config only mean_std-normalizes *pos* keys; rot6d stays raw.
            if key.endswith("pos"):
                parts.append(_normalize(raw, self._means[key], self._stds[key]))
            else:
                parts.append(np.asarray(raw, dtype=np.float32))
        s_t = np.concatenate(parts, axis=-1).astype(np.float32)
        if s_t.shape[-1] != POS_ROT_DIM:
            raise ValueError(f"s_t dim {s_t.shape[-1]} != {POS_ROT_DIM}")
        return torch.from_numpy(s_t).to(self.device)

    @torch.inference_mode()
    def select(
        self,
        actions_norm: torch.Tensor,
        s_t: torch.Tensor,
        pred_sf: torch.Tensor,
    ) -> Tuple[int, np.ndarray]:
        """Pick best of K normalized action chunks.

        Args:
            actions_norm: [K, H, D] (model action_pred space; may include pad dims)
            s_t: [D_posrot] or [1, D]
            pred_sf: [T, D_sf] or [1, T, D_sf] — use first POS_ROT_DIM cols
        Returns:
            best_index, scores [K]
        """
        if actions_norm.ndim != 3:
            raise ValueError(f"actions_norm expected [K,H,D], got {tuple(actions_norm.shape)}")
        k = min(self.bon_k, int(actions_norm.shape[0]))
        a = actions_norm[:k, : self.action_horizon, : self.action_dim].float().to(self.device)
        if s_t.ndim == 1:
            s_t = s_t.unsqueeze(0)
        s_t = s_t.float().to(self.device)[:, :POS_ROT_DIM]
        if pred_sf.ndim == 3:
            pred_sf = pred_sf[0]
        pred_sf = pred_sf.float().to(self.device)[:, :POS_ROT_DIM]
        s_b = s_t.expand(k, -1)
        sf_b = pred_sf.unsqueeze(0).expand(k, -1, -1)
        s_hat = self.wm(s_b, a)
        # distance() already restricts to SCORE_FRAME_SLICE (t+1..t+8).
        scores = (-self.crit.distance(s_hat, sf_b, mask=None)).detach().float().cpu().numpy()
        best = int(np.argmax(scores))
        return best, scores

    def next_seed_base(self, seed: int = 0) -> int:
        self._call_idx += 1
        return int(seed) + 100003 * self._call_idx
