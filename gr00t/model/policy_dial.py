# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1_dial import GR00T_N1_5_DIAL

COMPUTE_DTYPE = torch.bfloat16


def _confirm_loaded_weights(model, model_path) -> None:
    """Broadcast a length-1 null SF token and refuse a silent random init."""
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    weight_map = json.loads(index_path.read_text())["weight_map"]

    def _read(key: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(Path(model_path) / weight_map[key]), framework="pt") as handle:
            return handle.get_tensor(key)

    if hasattr(model, "null_sf_embed") and "null_sf_embed" in weight_map:
        saved = _read("null_sf_embed")
        dest = model.null_sf_embed
        if tuple(saved.shape) != tuple(dest.shape):
            if not (
                saved.ndim == 3
                and dest.ndim == 3
                and saved.shape[0] == 1
                and saved.shape[1] == 1
                and saved.shape[-1] == dest.shape[-1]
            ):
                raise RuntimeError(
                    f"null_sf_embed checkpoint {tuple(saved.shape)} cannot map onto {tuple(dest.shape)}"
                )
            saved = saved.expand(1, dest.shape[1], dest.shape[-1]).contiguous()
            with torch.no_grad():
                dest.copy_(saved.to(dtype=dest.dtype))
            print(f"[load] null_sf_embed broadcast to {tuple(dest.shape)}", flush=True)

    check_keys = ["bridge_projector.0.weight"]
    action_key = next(
        (key for key in weight_map if key.startswith("action_head.") and key.endswith("weight")),
        None,
    )
    if action_key is not None:
        check_keys.append(action_key)
    for key in check_keys:
        saved = _read(key)
        module = model
        for part in key.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        got = module.detach().float().cpu()
        if not torch.allclose(got, saved.float(), atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"{key} does not match the checkpoint")
    print("[load] checkpoint weights match", flush=True)


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class DialPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer_len: int = None,
        compute_bridge_loss: bool = False,
        guidance_scale: float = 1.0,
        async_vlm_inference: bool = False, # <-- New: whether to enable async
        vlm_update_period: int = 3,        # <-- New: VLM update frequency
        sf_eval_mode: str = "correct",
        sf_swap_bank_max: int = 64,
        wm_ckpt: Optional[str] = None,
        bon_k: int = 1,
        bon_seed: int = 0,
        bon_select: str = "score",
    ):
        """
        Initialize the DialPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
            sf_eval_mode: Online-SR SF ablation — correct | zero | swap.
            sf_swap_bank_max: Ring-buffer size for swap-mode wrong foresight.
            wm_ckpt: Optional frozen WM verifier .pt for Best-of-N action selection.
            bon_k: Number of action candidates to sample (1 = no BoN).
            bon_seed: Base seed for candidate resampling.
            bon_select: How to pick among K candidates — score (WM) | random.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path, tokenizer_len=tokenizer_len, compute_bridge_loss=compute_bridge_loss)
        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()
        self._guidance_scale = guidance_scale

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

        self.async_vlm_inference = async_vlm_inference
        self.vlm_update_period = vlm_update_period
        self._async_step = 0
        self._cached_backbone_feat = None

        # Online-SR SF ablation (correct | zero | swap). Applied in model.eval().
        self.sf_eval_mode = (sf_eval_mode or "correct").strip().lower()
        self.sf_swap_bank_max = int(sf_swap_bank_max)
        if hasattr(self.model, "set_sf_eval_mode"):
            self.model.set_sf_eval_mode(self.sf_eval_mode, bank_max=self.sf_swap_bank_max)
        elif self.sf_eval_mode != "correct":
            raise RuntimeError(
                f"sf_eval_mode={self.sf_eval_mode!r} requested but model has no set_sf_eval_mode"
            )

        self.bon_k = max(int(bon_k), 1)
        self.bon_seed = int(bon_seed)
        self.bon_select = (bon_select or "score").strip().lower()
        if self.bon_select not in ("score", "random"):
            raise ValueError(f"bon_select must be score|random, got {self.bon_select!r}")
        self._bon_call_idx = 0
        self._wm_bon = None
        if wm_ckpt:
            from gr00t.model.wm_verifier.online_bon import OnlineWMBoN

            self._wm_bon = OnlineWMBoN(
                wm_ckpt=wm_ckpt,
                policy_metadata=self.metadata,
                device=self.device,
                bon_k=self.bon_k,
            )
            print(
                f"[DialPolicy] WM-BoN enabled k={self.bon_k} select={self.bon_select} ckpt={wm_ckpt}",
                flush=True,
            )
        elif self.bon_k > 1 and self.bon_select == "score":
            raise ValueError("bon_k>1 with bon_select=score requires wm_ckpt")
        elif self.bon_k > 1:
            print(
                f"[DialPolicy] BoN sampling k={self.bon_k} select={self.bon_select} (no WM)",
                flush=True,
            )

    def set_sf_eval_mode(self, mode: str = "correct", bank_max: int | None = None) -> None:
        """Proxy to GR00T_N1_5_DIAL.set_sf_eval_mode for online-SR arms."""
        self.sf_eval_mode = (mode or "correct").strip().lower()
        if bank_max is not None:
            self.sf_swap_bank_max = int(bank_max)
        if not hasattr(self.model, "set_sf_eval_mode"):
            raise RuntimeError("loaded model has no set_sf_eval_mode")
        self.model.set_sf_eval_mode(self.sf_eval_mode, bank_max=self.sf_swap_bank_max)

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_input['guidance_scale'] = self._guidance_scale

        if self.bon_k > 1:
            s_t_for_bon = None
            if self._wm_bon is not None:
                s_t_for_bon = self._wm_bon.extract_s_t_from_normalized(normalized_input)
                if s_t_for_bon is None and self.bon_select == "score":
                    raise RuntimeError(
                        "WM-BoN could not read s_t from normalized state "
                        "(expected state[:, :18] wrist pos+rot6d)"
                    )
            elif self.bon_select == "score":
                raise RuntimeError("bon_select=score requires wm_ckpt")
            normalized_action = self._get_action_bon(normalized_input, s_t_for_bon)
        else:
            normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action, normalized_input)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _get_action_bon(
        self, normalized_input: Dict[str, Any], s_t: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Sample K action chunks; pick by WM score or random (same candidate seeds)."""

        def _clone(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, np.ndarray):
                return value.copy()
            if isinstance(value, dict):
                return {k: _clone(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_clone(v) for v in value]
            return value

        if self._wm_bon is not None:
            seed_base = self._wm_bon.next_seed_base(self.bon_seed)
        else:
            self._bon_call_idx += 1
            seed_base = int(self.bon_seed) + 100003 * self._bon_call_idx

        cands = []
        pred_sf = None
        backbone_feat = None  # cand0 full VLM; cand1..K-1 reuse (skip backbone)
        for i in range(self.bon_k):
            torch.manual_seed(seed_base + 10007 * (i + 1))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base + 10007 * (i + 1))
            copied = _clone(normalized_input)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                model_pred = self.model.get_action(
                    copied, backbone_features=backbone_feat
                )
            a = model_pred["action_pred"].float()
            if a.ndim == 3:
                a = a[0]
            cands.append(a.detach())
            # pred_sf / backbone are obs-conditioned (independent of action noise).
            if backbone_feat is None and model_pred.get("backbone_features") is not None:
                backbone_feat = model_pred["backbone_features"]
            if pred_sf is None and model_pred.get("pred_state_future") is not None:
                psf = model_pred["pred_state_future"].detach().float()
                if psf.ndim == 3:
                    psf = psf[0]
                pred_sf = psf
        stacked = torch.stack(cands, dim=0)  # [K,H,D]

        scores = None
        if self.bon_select == "score":
            if self._wm_bon is None or s_t is None:
                raise RuntimeError("bon_select=score requires WM and s_t")
            if pred_sf is None:
                raise RuntimeError("WM-BoN needs pred_state_future from policy get_action")
            best, scores = self._wm_bon.select(stacked, s_t, pred_sf)
            tag = "score"
        else:
            # Deterministic among the same K pool (reproducible random arm).
            rng = np.random.RandomState(seed_base + 17)
            best = int(rng.randint(0, self.bon_k))
            tag = "random"
            if self._wm_bon is not None and s_t is not None and pred_sf is not None:
                _, scores = self._wm_bon.select(stacked, s_t, pred_sf)

        if scores is not None:
            print(
                f"[WM-BoN] select={tag} pick={best}/{self.bon_k} "
                f"scores={np.array2string(scores, precision=4)}",
                flush=True,
            )
        else:
            print(f"[WM-BoN] select={tag} pick={best}/{self.bon_k}", flush=True)
        return stacked[best].unsqueeze(0)

    def propose_bon_with_meta(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Sample K BoN candidates and return scores + unnormalized action dicts.

        Used by same-state branch probes: caller save/restores sim and executes
        each candidate chunk to compare verifier ranking vs real outcomes.
        """
        if self.bon_k <= 1:
            raise RuntimeError("propose_bon_with_meta requires bon_k > 1")

        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_input["guidance_scale"] = self._guidance_scale

        s_t = None
        if self._wm_bon is not None:
            s_t = self._wm_bon.extract_s_t_from_normalized(normalized_input)

        def _clone(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, np.ndarray):
                return value.copy()
            if isinstance(value, dict):
                return {k: _clone(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_clone(v) for v in value]
            return value

        if self._wm_bon is not None:
            seed_base = self._wm_bon.next_seed_base(self.bon_seed)
        else:
            self._bon_call_idx += 1
            seed_base = int(self.bon_seed) + 100003 * self._bon_call_idx

        cands = []
        pred_sf = None
        backbone_feat = None  # cand0 full VLM; cand1..K-1 reuse (skip backbone)
        for i in range(self.bon_k):
            torch.manual_seed(seed_base + 10007 * (i + 1))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base + 10007 * (i + 1))
            copied = _clone(normalized_input)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                model_pred = self.model.get_action(
                    copied, backbone_features=backbone_feat
                )
            a = model_pred["action_pred"].float()
            if a.ndim == 3:
                a = a[0]
            cands.append(a.detach())
            if backbone_feat is None and model_pred.get("backbone_features") is not None:
                backbone_feat = model_pred["backbone_features"]
            if pred_sf is None and model_pred.get("pred_state_future") is not None:
                psf = model_pred["pred_state_future"].detach().float()
                if psf.ndim == 3:
                    psf = psf[0]
                pred_sf = psf
        stacked = torch.stack(cands, dim=0)

        scores = None
        scores_pos = None
        scores_rot = None
        best = 0
        if self._wm_bon is not None and s_t is not None and pred_sf is not None:
            from gr00t.model.wm_verifier.action_conditioned_dynamics import POS_ROT_DIM

            wm = self._wm_bon
            k = min(self.bon_k, int(stacked.shape[0]))
            a = stacked[:k, : wm.action_horizon, : wm.action_dim].float().to(wm.device)
            st = s_t.float().to(wm.device)
            if st.ndim == 1:
                st = st.unsqueeze(0)
            st = st[:, :POS_ROT_DIM].expand(k, -1)
            sf = pred_sf.float().to(wm.device)
            if sf.ndim == 3:
                sf = sf[0]
            sf = sf[:, :POS_ROT_DIM].unsqueeze(0).expand(k, -1, -1)
            s_hat = wm.wm(st, a)
            # Per-sample pos/rot distances on score window (same as distance()).
            crit = wm.crit
            sl = crit.score_slice
            p = s_hat[:, sl]
            g = sf[:, sl]
            m = torch.ones(*p.shape[:2], 1, device=p.device, dtype=p.dtype)
            w = m.sum(dim=(1, 2)).clamp_min(1.0)
            std = crit.dim_std()

            def _one(idx: torch.Tensor) -> torch.Tensor:
                gstd = std.index_select(0, idx).view(1, 1, -1)
                err = (p.index_select(-1, idx) - g.index_select(-1, idx)) / gstd
                return (err.square() * m).sum(dim=(1, 2)) / (w * idx.numel())

            d_pos = _one(crit.pos_idx)
            d_rot = _one(crit.rot_idx)
            d_tot = d_pos + d_rot
            scores = (-d_tot).detach().float().cpu().numpy()
            scores_pos = (-d_pos).detach().float().cpu().numpy()
            scores_rot = (-d_rot).detach().float().cpu().numpy()
            best = int(np.argmax(scores))
            s_hat_np = s_hat.detach().float().cpu().numpy()
            pred_sf_np = sf[0].detach().float().cpu().numpy()
            s_t_np = st[0].detach().float().cpu().numpy()
        else:
            best = 0
            s_hat_np = None
            pred_sf_np = None
            s_t_np = s_t.detach().float().cpu().numpy() if s_t is not None else None

        # Random index independent of scores, but tied to the same seed_base
        # so score/random share the identical candidate pool.
        rng = np.random.RandomState(int(seed_base) + 17)
        random_pick = int(rng.randint(0, self.bon_k))

        # Mean-proximity / smoothness diagnostics on normalized actions.
        stacked_np = stacked.detach().float().cpu().numpy()
        mean_a = stacked_np.mean(axis=0, keepdims=True)
        dist_to_mean = np.sqrt(((stacked_np - mean_a) ** 2).mean(axis=(1, 2)))
        # Temporal smoothness: mean L2 of consecutive action deltas.
        if stacked_np.shape[1] > 1:
            deltas = stacked_np[:, 1:] - stacked_np[:, :-1]
            smoothness = np.sqrt((deltas**2).mean(axis=(1, 2)))
        else:
            smoothness = np.zeros((stacked_np.shape[0],), dtype=np.float32)
        # Pairwise candidate dispersion (mean pairwise L2 over flattened chunk).
        flat = stacked_np.reshape(stacked_np.shape[0], -1)
        if flat.shape[0] > 1:
            diffs = flat[:, None, :] - flat[None, :, :]
            pairwise = np.sqrt((diffs**2).sum(axis=-1))
            # upper triangle mean
            iu = np.triu_indices(flat.shape[0], k=1)
            cand_dispersion = float(pairwise[iu].mean()) if iu[0].size else 0.0
        else:
            cand_dispersion = 0.0

        unnorm_list = []
        for i in range(self.bon_k):
            un = self._get_unnormalized_action(stacked[i].unsqueeze(0), normalized_input)
            if not is_batch:
                un = squeeze_dict_values(un)
            unnorm_list.append(un)

        return {
            "best": best,  # score pick
            "random_pick": random_pick,  # independent of scores; same pool
            "scores": scores,
            "scores_pos": scores_pos,
            "scores_rot": scores_rot,
            "dist_to_mean": dist_to_mean,
            "smoothness": smoothness,
            "cand_dispersion": cand_dispersion,
            "actions_norm": stacked_np,
            "actions_unnorm": unnorm_list,
            "seed_base": seed_base,
            "s_t": s_t_np,  # current wrist pos+rot6d (normalized)
            "s_hat": s_hat_np,  # WM dynamics pred [K, T, 18]
            "pred_sf": pred_sf_np,  # s* foresight target [T, 18]
        }

    def get_action_and_bridgeloss(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        # print(f"normalized_input.keys: {normalized_input.keys()}")
        # print(f"normalized_input['orig_state'].keys: {normalized_input['orig_state'].keys()}")
        normalized_input['guidance_scale'] = self._guidance_scale

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)
        normalized_action = model_pred["action_pred"].float()

        unnormalized_action = self._get_unnormalized_action(normalized_action, normalized_input)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)



        bridge_loss = model_pred.get('bridge_loss', None)
        if bridge_loss is not None:
            # bridge_loss = {'bridge_loss': bridge_loss.cpu().numpy()}
            # if not is_batch:
            #     bridge_loss = squeeze_dict_values(bridge_loss)
            bridge_loss = np.squeeze(bridge_loss.cpu().numpy(), axis=0)
            unnormalized_action['bridge_loss'] = bridge_loss

        return unnormalized_action



    def get_action_and_debug_features(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)

        # 2. Extract the processed current frame (Current RGB)
        # Typically the key is 'vlm_pixel_values' [B, T, C, H, W]
        obs_tensors = normalized_input['vlm_pixel_values']
        # print(f"obs_tensors.shape: {obs_tensors.shape}")
        # curr_rgb = self._tensor_to_rgb(obs_tensors[0, -1]) # Batch 0, last frame
        # print(f"normalized_input.keys: {normalized_input.keys()}")

        # 2. Find the dynamic grid size (note: Qwen typically uses these names)
        grid_thw = normalized_input['vlm_image_grid_thw'][0]

        # 3. Restore RGB
        curr_rgb = self._tensor_to_rgb_from_qwen(obs_tensors, grid_thw)

        # 3. Extract the processed goal frame (Goal RGB)
        # Depending on your config, the key may be 'goal_image_pixel_values' or 'act_goal_image_pixel_values'
        # This Tensor has already gone through the same Resize and Crop

        if 'goal_image_pixel_values' in normalized_input:
            goal_tensors = normalized_input['goal_image_pixel_values']
            # goal_rgb = self._tensor_to_rgb(goal_tensors[0, -1])
            # Note: goal image may have its own grid_thw
            goal_grid_thw = normalized_input.get('goal_image_image_grid_thw', grid_thw.unsqueeze(0))[0]
            goal_rgb = self._tensor_to_rgb_from_qwen(goal_tensors, goal_grid_thw)
        else:
            # If not in inputs, use original image as placeholder (but recommended to include it)
            goal_rgb = curr_rgb 

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            # We can get features by passing params to model.forward or by directly calling submodules
            # Here we assume the model forward has been modified to return features, or we manually call at policy level
            backbone_inputs, action_inputs = self.model.prepare_input(normalized_input)
            
            # 1. Get Backbone output (containing predicted subgoal features)
            backbone_outputs = self.model.backbone(backbone_inputs)
            
            # 2. Replicate model.forward internal logic to get Obs and GT-Goal features
            batch_size = backbone_inputs['state'].shape[0]
            
            # Predicted Subgoal features (after Projector)
            # Note: based on the code logic, features are at [:, -num_bridge_tokens:]
            raw_pred_feat = backbone_outputs['backbone_features'][:, -self.model.num_bridge_tokens:]
            # pred_goal_feat = self.model.bridge_projector(raw_pred_feat)
            if getattr(self.model, 'use_separate_projector_for_loss', False):
                pred_goal_feat = self.model.bridge_projector_for_loss(raw_pred_feat)
            else:
                pred_goal_feat = self.model.bridge_projector(raw_pred_feat)
            
            # Get Current Obs features
            vlm_prefix = "vlm_"
            obs_input = {k.removeprefix(vlm_prefix): v for k, v in backbone_inputs.items() if k.startswith(vlm_prefix)}
            obs_input.pop("image_sizes", None); obs_input.pop("input_ids", None); obs_input.pop("attention_mask", None)
            obs_feat = self.model.bridge_vision_model(*obs_input.values())
            # Reshape to [B, 64, D]
            obs_feat = obs_feat.reshape(batch_size, -1, self.model.num_bridge_tokens, obs_feat.shape[-1])[:, -1]

            # Get GT Subgoal features
            goal_image_prefix = "goal_image_"
            goal_input = {k.removeprefix(goal_image_prefix): v for k, v in backbone_inputs.items() if k.startswith(goal_image_prefix)}
            gt_goal_feat = self.model.bridge_goal_model(*goal_input.values())
            gt_goal_feat = gt_goal_feat.reshape(batch_size, -1, gt_goal_feat.shape[-1])

            # Normal Action Head execution to get actions
            # Construct bridge_outputs for action_head (simplified, reusing forward logic)
            normalized_input['guidance_scale'] = self._guidance_scale
            res = self.model.forward(normalized_input, action_mode=True)
            
        # Return results
        debug_data = {
            "obs_feat": obs_feat[0].float().detach().cpu().numpy(),        # [64, D]
            "gt_goal_feat": gt_goal_feat[0].float().detach().cpu().numpy(), # [64, D]
            "pred_goal_feat": pred_goal_feat[0].float().detach().cpu().numpy(), # [64, D]
            "curr_rgb": curr_rgb,   # current image
            "goal_rgb": goal_rgb    # goal image
        }
        
        # Also include the action results
        unnormalized_action = self._get_unnormalized_action(res["action_pred"].float(), normalized_input)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        bridge_loss = res.get('bridge_loss', None)
        if bridge_loss is not None:
            bridge_loss = np.squeeze(bridge_loss.cpu().numpy(), axis=0)
            unnormalized_action['bridge_loss'] = bridge_loss
        
        return unnormalized_action, debug_data

    # def _tensor_to_rgb(self, tensor):
    #     """Helper: restore model input Tensor to visualizable RGB"""
    #     print(f"tensor.shape: {tensor.shape} !!!")
    #     img = tensor.permute(1, 2, 0).cpu().float().numpy()
    #     # Simple inverse normalization
    #     img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    #     return (img * 255).astype(np.uint8)


    def _tensor_to_rgb_from_qwen(self, tensor, grid_thw):
        """
        Accurately restore based on Qwen2VLImageProcessor source code logic.
        tensor: [L, 1176]
        grid_thw: [grid_t, grid_h, grid_w] -> e.g., [1, 16, 16]
        """
        grid_t, grid_h, grid_w = grid_thw.cpu().numpy()
        
        # Constants defined in source code
        patch_size = 14
        temporal_patch_size = 2
        merge_size = 2
        channel = 3

        # 1. Inverse reshape: reverse of the source code flatten_patches reshape
        # Logic: decompose D (1176) back to the post-transpose order in source code
        # Order: grid_t, grid_h // merge_size, grid_w // merge_size, merge_size, merge_size, channel, temporal_patch_size, patch_size, patch_size
        x = tensor.view(
            grid_t,
            grid_h // merge_size,
            grid_w // merge_size,
            merge_size,
            merge_size,
            channel,
            temporal_patch_size,
            patch_size,
            patch_size,
        )

        # 2. Inverse transpose: reverse of source code patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        # Source index mapping: 
        # 0->0(t), 1->3(h_maj), 2->6(w_maj), 3->4(h_min), 4->7(w_min), 5->2(c), 6->1(t_patch), 7->5(h_patch), 8->8(w_patch)
        # We want to restore to: [grid_t, temporal_patch_size, channel, h_major, h_minor, h_patch, w_major, w_minor, w_patch]
        # Corresponding current indices in x: (0, 6, 5, 1, 3, 7, 2, 4, 8)
        x = x.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).contiguous()

        # 3. Restore to image shape [T, C, H, W]
        # Here H = (grid_h // merge_size) * merge_size * patch_size = grid_h * patch_size
        x = x.view(
            grid_t * temporal_patch_size,
            channel,
            grid_h * patch_size,
            grid_w * patch_size
        )

        # 4. Post-processing: take the last frame and convert
        # Reshape to [H, W, C]
        img = x[-1].permute(1, 2, 0).detach().cpu().float().numpy()

        # 5. Inverse normalization (using OPENAI_CLIP_MEAN / STD from source code)
        # Corresponding to [0.48145466, 0.4578275, 0.40821073] and [0.26862954, 0.26130258, 0.27577711]
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        return (img * 255).astype(np.uint8)

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # --- Async inference core logic ---
        backbone_feat_to_use = None
        
        if self.async_vlm_inference:
            # If not an update step and cache exists, use cached features
            if (self._async_step % self.vlm_update_period != 0) and (self._cached_backbone_feat is not None):
                # print("update!")
                backbone_feat_to_use = self._cached_backbone_feat
        # -----------------------


        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(
                normalized_input,
                backbone_features=backbone_feat_to_use
            )

        # --- Update async state ---
        if self.async_vlm_inference:
            self._cached_backbone_feat = model_pred.get("backbone_features")
            self._async_step += 1
            # print(self._async_step)
        # -------------------

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor, normalized_input: torch.Tensor) -> Dict[str, Any]:
        # return self.unapply_transforms({"action": normalized_action.cpu()})
        data_to_unapply = normalized_input.copy()
        data_to_unapply["action"] = normalized_action.cpu()

        for k, v in data_to_unapply.items():
            if isinstance(v, torch.Tensor):
                data_to_unapply[k] = v.to(torch.float64)
            elif isinstance(v, np.ndarray):
                # If it's a numpy array, also convert to float32 tensor,
                # since subsequent Transform matrix operations require tensors
                data_to_unapply[k] = torch.from_numpy(v).to(torch.float64)

        unnormalized_data = self.unapply_transforms(data_to_unapply)
        action_out = {k: v for k, v in unnormalized_data.items() if k.startswith("action.")}
        return action_out

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path, tokenizer_len=None, compute_bridge_loss=False):
        # print("loading model ...")
        model = GR00T_N1_5_DIAL.from_pretrained(
            model_path, 
            tokenizer_len=tokenizer_len,
            torch_dtype=COMPUTE_DTYPE,
            compute_bridge_loss=compute_bridge_loss,
            ignore_mismatched_sizes=True,
        )
        _confirm_loaded_weights(model, model_path)
        model.eval()  # Set model to eval mode
        # exit()

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHead class
            from gr00t.model.action_head.flow_matching_action_head_dial import (
                FlowmatchingActionHeadDIAL,
            )

            # Create new action head with updated config
            new_action_head = FlowmatchingActionHeadDIAL(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore

        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data
