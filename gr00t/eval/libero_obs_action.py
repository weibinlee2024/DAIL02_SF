"""
LIBERO online-eval helpers for DIAL (same suites as RND dense collection).

Maps:
  LIBERO obs  -> DialPolicy dict keys (video.image / wrist_image / state.* / language)
  DialPolicy action.* -> 7D env action (delta xyz + delta rpy + gripper)

Full gym loop should reuse VLA-RFT `run_libero_eval` with DialPolicy backend, or
call these mappers from a thin client. Default n_action_steps=8 (locked).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

LIBERO_SUITE_NAMES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

# Locked: align RND collect chunk + LiberoDataConfig.action_indices
DEFAULT_N_ACTION_STEPS = 8


def _as_uint8_hwc(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    return arr


def axis_angle_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (4,) xyzw to axis-angle (3,)."""
    try:
        from scipy.spatial.transform import Rotation as Rot

        return Rot.from_quat(np.asarray(quat, dtype=np.float64)).as_rotvec().astype(np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def proprio_from_libero_obs(obs: Mapping[str, Any]) -> np.ndarray:
    """Build 8D proprio: eef_pos(3) + eef_ori_aa(3) + gripper(2)."""
    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]
        aa = axis_angle_from_quat_xyzw(np.asarray(obs["robot0_eef_quat"]).reshape(-1)[:4])
        grip = np.asarray(obs.get("robot0_gripper_qpos", [0.0, 0.0]), dtype=np.float32).reshape(-1)
        if grip.size == 1:
            grip = np.array([grip[0], grip[0]], dtype=np.float32)
        return np.concatenate([pos, aa, grip[:2]]).astype(np.float32)

    if "ee_states" in obs:
        ee = np.asarray(obs["ee_states"], dtype=np.float32).reshape(-1)
        grip = np.asarray(obs.get("gripper_states", [0.0, 0.0]), dtype=np.float32).reshape(-1)
        if grip.size == 1:
            grip = np.array([grip[0], grip[0]], dtype=np.float32)
        return np.concatenate([ee[:6], grip[:2]]).astype(np.float32)

    raise KeyError(f"Cannot build proprio from obs keys={list(obs.keys())}")


def libero_obs_to_dial(
    obs: Mapping[str, Any],
    language: str,
    *,
    add_time_dim: bool = True,
) -> Dict[str, Any]:
    """
    Convert a raw LIBERO observation dict into DialPolicy / ZMQ payload keys.

    Expected video shapes without batch: (T,H,W,C) with T=1 when add_time_dim=True.
    """
    agent = obs.get("agentview_image") or obs.get("agentview_rgb") or obs.get("image")
    wrist = (
        obs.get("robot0_eye_in_hand_image")
        or obs.get("eye_in_hand_rgb")
        or obs.get("wrist_image")
    )
    if agent is None or wrist is None:
        raise KeyError(f"Missing cameras in obs keys={list(obs.keys())}")

    agent = _as_uint8_hwc(agent)
    wrist = _as_uint8_hwc(wrist)
    proprio = proprio_from_libero_obs(obs)

    def _t(x: np.ndarray) -> np.ndarray:
        return x[None, ...] if add_time_dim else x

    return {
        "video.image": _t(agent),
        "video.wrist_image": _t(wrist),
        "state.eef_position": _t(proprio[0:3]),
        "state.eef_rotation": _t(proprio[3:6]),
        "state.gripper_position": _t(proprio[6:8]),
        "annotation.human.action.task_description": language,
    }


def dial_action_to_libero_7d(action_dict: Mapping[str, np.ndarray], step: int = 0) -> np.ndarray:
    """
    Pack DialPolicy action chunk keys into a single 7D env action at `step`.

    Accepts shapes (H, D) or (1, H, D).
    """

    def _slice(key: str, dim: int) -> np.ndarray:
        if key not in action_dict:
            raise KeyError(f"Missing action key {key}; got {list(action_dict.keys())}")
        arr = np.asarray(action_dict[key], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"{key} expected (H,D), got {arr.shape}")
        return arr[step, :dim]

    pos = _slice("action.eef_position_delta", 3)
    rot = _slice("action.eef_rotation_delta", 3)
    grip = _slice("action.gripper_position", 1)
    return np.concatenate([pos, rot, grip]).astype(np.float32)


def dial_action_chunk_to_libero(
    action_dict: Mapping[str, np.ndarray],
    n_action_steps: int = DEFAULT_N_ACTION_STEPS,
) -> np.ndarray:
    """Return (H, 7) env actions from DialPolicy outputs."""
    sample = next(iter(action_dict.values()))
    arr = np.asarray(sample)
    horizon = arr.shape[-2] if arr.ndim >= 2 else n_action_steps
    h = min(int(horizon), int(n_action_steps))
    return np.stack([dial_action_to_libero_7d(action_dict, step=i) for i in range(h)], axis=0)
