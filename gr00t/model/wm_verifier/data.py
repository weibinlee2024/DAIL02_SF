"""Dataset helpers for WM-as-Verifier (no VLM / no main model)."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from gr00t.data.dataset import (
    LeRobotSingleDatasetWithGoalImage,
    MultiEmbodimentLeRobotMixtureDataset,
)
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.experiment.data_config_dial import (
    resolve_state_future_keys,
    load_data_config,
)
from gr00t.model.wm_verifier.action_conditioned_dynamics import POS_ROT_DIM


class WMPackTransform(ModalityTransform):
    """Pack normalized action + state_future after ConcatTransform (no VLM)."""

    apply_to: list[str] = []
    action_horizon: int = 16
    state_future_horizon: int = 16  # H; frames = H+1
    state_future_concat_order: list[str] = []
    state_future_dim: int = 30
    pos_rot_dim: int = POS_ROT_DIM

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        action = data["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        assert action.shape[0] == self.action_horizon, (
            f"action T={action.shape[0]} != action_horizon={self.action_horizon}"
        )
        n_action_dims = int(action.shape[1])
        action_mask = np.ones((self.action_horizon, n_action_dims), dtype=bool)

        expected_len = self.state_future_horizon + 1
        frames = []
        for k in self.state_future_concat_order:
            if k not in data:
                raise KeyError(f"missing {k} for state_future pack")
            v = data[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            frames.append(np.asarray(v, dtype=np.float32))
        state_future = np.concatenate(frames, axis=-1)
        assert state_future.shape[0] == expected_len, (
            f"state_future T={state_future.shape[0]} != {expected_len}"
        )
        if state_future.shape[-1] != self.state_future_dim:
            raise ValueError(
                f"state_future dim {state_future.shape[-1]} != {self.state_future_dim}"
            )
        time_mask = data.get("state_future_time_mask")
        if time_mask is None:
            raise ValueError("state_future_time_mask missing")
        if isinstance(time_mask, torch.Tensor):
            time_mask = time_mask.detach().cpu().numpy()
        time_mask = np.asarray(time_mask).astype(bool).reshape(-1)
        assert time_mask.shape[0] == expected_len
        sf_mask = np.broadcast_to(time_mask[:, None], state_future.shape).copy()

        # Verifier target: pos+rot6d only (drop hand).
        s_full = state_future[:, : self.pos_rot_dim]
        s_mask = sf_mask[:, : self.pos_rot_dim]
        s_t = s_full[0].copy()

        return {
            "s_t": s_t.astype(np.float32),
            "a_chunk": action.astype(np.float32),
            "a_mask": action_mask,
            "s_future": s_full.astype(np.float32),
            "s_mask": s_mask[:, 0].astype(bool),  # [T] temporal
        }


def _build_wm_transforms(data_config_cls) -> ComposedModalityTransform:
    """Reuse data_config state/action/SF normalization; skip video + VLM."""
    src_keys = data_config_cls.get_state_future_source_keys()
    sf_keys = data_config_cls.get_state_future_concat_order()
    sf_horizon = (
        data_config_cls.state_future_horizon
        if getattr(data_config_cls, "state_future_horizon", None) is not None
        else len(data_config_cls.action_indices)
    )
    joint_keys = [
        "state.right_arm",
        "state.right_hand",
        "state.left_arm",
        "state.left_hand",
        "state.waist",
    ]
    joint_keys = [k for k in joint_keys if k in data_config_cls.state_keys]
    transforms = [
        StateActionToTensor(apply_to=data_config_cls.state_keys),
        StateActionSinCosTransform(apply_to=joint_keys) if joint_keys else None,
        StateActionTransform(
            apply_to=data_config_cls.state_keys,
            normalization_modes=data_config_cls.state_normalization_modes,
        ),
        StateActionToTensor(apply_to=data_config_cls.action_keys),
        StateActionTransform(
            apply_to=data_config_cls.action_keys,
            normalization_modes=data_config_cls.action_normalization_modes,
        ),
        *data_config_cls._build_state_future_pipeline(),
        ConcatTransform(
            video_concat_order=[],
            state_concat_order=data_config_cls.state_keys,
            action_concat_order=data_config_cls.action_keys,
        ),
        WMPackTransform(
            action_horizon=len(data_config_cls.action_indices),
            state_future_horizon=sf_horizon,
            state_future_concat_order=sf_keys,
            state_future_dim=data_config_cls.get_state_future_dim(),
            pos_rot_dim=POS_ROT_DIM,
        ),
    ]
    transforms = [t for t in transforms if t is not None]
    return ComposedModalityTransform(transforms=transforms)


def _walk_transforms(transform):
    children = getattr(transform, "transforms", None)
    if children:
        for child in children:
            yield from _walk_transforms(child)
    else:
        yield transform


def normalization_snapshot(transforms) -> dict:
    """Mean/std actually installed on the state/action normalizers."""
    keys = {}
    for transform in _walk_transforms(transforms):
        stats = getattr(transform, "normalization_statistics", None) or {}
        modes = getattr(transform, "normalization_modes", None) or {}
        if not isinstance(stats, dict):
            continue
        for key, st in stats.items():
            if not isinstance(st, dict):
                continue
            mean = st.get("mean")
            std = st.get("std")
            keys[key] = {
                "mode": modes.get(key),
                "mean": None if mean is None else [round(float(x), 8) for x in mean],
                "std": None if std is None else [round(float(x), 8) for x in std],
            }
    blob = json.dumps(keys, sort_keys=True)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return {"keys": keys, "fingerprint": digest}


def lock_norm_to_policy(dataset, policy_path: str) -> dict:
    """Overwrite mixture stats with the frozen policy checkpoint metadata.

    Dataset init computes per-task stats and the mixture then overwrites the
    shared transforms with a weighted average. Scoring and WM training must
    both use the policy's experiment_cfg/metadata.json instead.
    """
    from gr00t.data.dataset import load_metadata

    policy_path = os.path.realpath(policy_path)
    metadatas = load_metadata(policy_path)
    datasets = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
    for ds in datasets:
        metadata = metadatas.get(ds.tag)
        if metadata is None:
            raise KeyError(
                f"policy metadata has no embodiment tag {ds.tag!r} ({policy_path})"
            )
        ds.set_transforms_metadata(metadata)
    if hasattr(dataset, "merged_metadata"):
        dataset.merged_metadata = metadatas
    snap = normalization_snapshot(datasets[0].transforms)
    return {
        "norm_source": policy_path,
        "norm_fingerprint": snap["fingerprint"],
        "norm_snapshot": snap["keys"],
        "stats_version": 2,
    }


def build_wm_train_dataset(
    dataset_paths: list[str],
    data_config: str = "fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop",
    data_split: str = "[:-10]",
    embodiment_tag: str = "gr1",
    state_future_dim: int = 30,
    video_backend: str = "decord",
    policy_path: Optional[str] = None,
):
    """Same offline tasks as DIAL; labels from state_future / action transforms."""
    os.environ["SF_FORCE_KEY_DIM"] = "30" if state_future_dim != 60 else "60"
    data_config_cls = load_data_config(
        data_config, use_bridge=False, ignore_lang_prefix=True
    )
    data_config_cls.use_state_future = True
    data_config_cls.state_future_horizon = len(data_config_cls.action_indices)
    data_config_cls.state_future_keys = resolve_state_future_keys(state_future_dim)

    # Drop video/language so the loader never touches images / tokenizer.
    modality_configs = data_config_cls.modality_config()
    modality_configs = {
        k: v for k, v in modality_configs.items() if k in ("state", "action", "state_future")
    }
    transforms = _build_wm_transforms(data_config_cls)

    singles = []
    for p in dataset_paths:
        if not os.path.isdir(p):
            raise FileNotFoundError(p)
        singles.append(
            LeRobotSingleDatasetWithGoalImage(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=video_backend,
                split=data_split,
            )
        )
    if len(singles) == 1:
        mixed = singles[0]
    else:
        mixed = MultiEmbodimentLeRobotMixtureDataset(
            data_mixture=[(ds, 1.0) for ds in singles],
            mode="train",
            balance_dataset_weights=True,
            balance_trajectory_weights=True,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
    meta = {
        "action_horizon": len(data_config_cls.action_indices),
        "state_horizon": data_config_cls.state_future_horizon + 1,
        "state_future_dim": data_config_cls.get_state_future_dim(),
        "pos_rot_dim": POS_ROT_DIM,
        "data_config": data_config,
        "state_keys": list(data_config_cls.state_future_keys),
        "action_delta_indices": list(modality_configs["action"].delta_indices),
        "state_future_delta_indices": list(modality_configs["state_future"].delta_indices),
        "data_split": data_split,
    }
    if policy_path:
        # Must run after mixture init, which otherwise installs its own stats.
        meta.update(lock_norm_to_policy(mixed, policy_path))
    return mixed, meta


def wm_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    s_t = torch.stack([torch.as_tensor(b["s_t"], dtype=torch.float32) for b in batch])
    a = torch.stack([torch.as_tensor(b["a_chunk"], dtype=torch.float32) for b in batch])
    s_f = torch.stack([torch.as_tensor(b["s_future"], dtype=torch.float32) for b in batch])
    s_m = torch.stack([torch.as_tensor(b["s_mask"], dtype=torch.bool) for b in batch])
    return {"s_t": s_t, "a_chunk": a, "s_future": s_f, "s_mask": s_m}


def infer_action_dim(dataset: Dataset, max_probe: int = 8) -> int:
    """Action width after concat (no pad); used to size the MLP."""
    for i in range(min(max_probe, len(dataset))):
        sample = dataset[i]
        a = np.asarray(sample["a_chunk"])
        if a.ndim == 2 and a.shape[1] > 0:
            return int(a.shape[1])
    raise RuntimeError("could not infer action_dim from dataset")
