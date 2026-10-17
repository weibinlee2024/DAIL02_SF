#!/usr/bin/env python3
"""(4) Offline WM-as-Verifier score validity — before any online experiment.

Freeze policy, sample K∈{4,8} action candidates, score = -D(ŝ, s*) on t+1..t+8.
Report per K:
  - Δ action_mse = mean_cand_mse - top1_mse   (positive ⇒ score picks closer actions)
  - pairwise accuracy (score order matches action_mse order)
  - Spearman(score, -action_mse)

Does not run simulation. Requires a trained WM ckpt and a frozen policy ckpt.
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tyro
from scipy.stats import spearmanr
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gr00t.data.dataset import LeRobotSingleDatasetWithGoalImage
from gr00t.experiment.data_config_dial import load_data_config, resolve_state_future_keys
from gr00t.model.policy_dial import DialPolicy, unsqueeze_dict_values
from gr00t.model.transforms import BRIDGE_TOKENS
from gr00t.model.wm_verifier.action_conditioned_dynamics import (
    POS_ROT_DIM,
    ActionConditionedDynamics,
    GroupNormMSE,
)
from gr00t.model.wm_verifier.data import _build_wm_transforms

DEFAULT_TASKS = [
    "PnPCupToDrawerClose",
    "PnPCanToDrawerClose",
    "PnPBottleToCabinetClose",
]


@dataclass
class Args:
    policy_path: str
    """Frozen System-1/2 checkpoint (DialPolicy)."""
    wm_ckpt: str
    """WM verifier checkpoint-*.pt from train_wm_verifier.py."""
    data_root: str = "/home/weibin02/DIAL_DATASETS/LeRobot-AugPosRot-Correct"
    dataset_path: Optional[List[str]] = None
    data_config: str = "fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop"
    data_split: str = "[-10:]"
    """Holdout split for offline validity (not the WM train split)."""
    embodiment_tag: str = "gr1"
    ks: List[int] = field(default_factory=lambda: [4, 8])
    trajs: int = 4
    points: int = 4
    action_horizon: int = 16
    denoising_steps: int = 4
    seed: int = 0
    device: str = "cuda"
    save_json: Optional[str] = None
    video_backend: str = "decord"


def _pairwise_accuracy(scores: np.ndarray, mses: np.ndarray) -> float:
    """Fraction of pairs where higher score ↔ lower action_mse."""
    n = len(scores)
    if n < 2:
        return float("nan")
    ok = 0
    tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(mses[i] - mses[j]) < 1e-12:
                continue
            tot += 1
            # Prefer lower mse; score should be higher for better cand
            score_pref_i = scores[i] > scores[j]
            mse_pref_i = mses[i] < mses[j]
            if score_pref_i == mse_pref_i:
                ok += 1
    return float(ok / tot) if tot else float("nan")


def _spearman_score_neg_mse(scores: np.ndarray, mses: np.ndarray) -> float:
    if len(scores) < 3 or np.std(scores) < 1e-12 or np.std(mses) < 1e-12:
        return float("nan")
    rho, _ = spearmanr(scores, -mses)
    return float(rho)


def _load_wm(wm_ckpt: str, device: torch.device):
    ckpt = torch.load(wm_ckpt, map_location="cpu")
    meta = ckpt.get("meta") or {}
    args_saved = ckpt.get("args") or {}
    model = ActionConditionedDynamics(
        state_dim=POS_ROT_DIM,
        action_dim=int(meta["action_dim"]),
        action_horizon=int(meta["action_horizon"]),
        state_horizon=int(meta["state_horizon"]),
        hidden_dim=int(args_saved.get("hidden_dim", 512)),
        num_layers=int(args_saved.get("num_layers", 3)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    crit = GroupNormMSE(state_dim=POS_ROT_DIM)
    if "crit" in ckpt:
        crit.load_state_dict(ckpt["crit"])
    elif "pos_std" in (ckpt.get("meta") or {}):
        # fallback if only meta stds were saved in wm_meta.json path
        pass
    crit.to(device).eval()
    return model, crit, meta


def _resolve_paths(args: Args) -> list[str]:
    if args.dataset_path:
        return list(args.dataset_path)
    return [os.path.join(args.data_root, f"gr1_unified.{t}") for t in DEFAULT_TASKS]


@torch.no_grad()
def _sample_candidates(
    policy: DialPolicy,
    obs: dict,
    k: int,
    action_dim: int,
    seed_base: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Return normalized a_chunk [K,H,D], per-cand action_mse vs GT, and a_gt."""
    obs_copy = {key: (v if isinstance(v, np.ndarray) else np.array(v)) for key, v in obs.items()}
    if not policy._check_state_is_batched(obs_copy):
        obs_copy = unsqueeze_dict_values(obs_copy)
    normalized_input = policy.apply_transforms(obs_copy)
    a_gt = normalized_input["action"]
    if isinstance(a_gt, torch.Tensor):
        a_gt_np = a_gt.detach().float().cpu().numpy()
    else:
        a_gt_np = np.asarray(a_gt, dtype=np.float32)
    if a_gt_np.ndim == 3:
        a_gt_np = a_gt_np[0]
    a_gt_np = a_gt_np[:, :action_dim]

    cands = []
    mses = []
    for i in range(k):
        torch.manual_seed(seed_base + 10007 * (i + 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_base + 10007 * (i + 1))
        a_norm = policy._get_action_from_normalized_input(normalized_input)
        a = a_norm.detach().float().cpu().numpy()
        if a.ndim == 3:
            a = a[0]
        a = a[:, :action_dim]
        cands.append(a)
        mses.append(float(np.mean((a - a_gt_np) ** 2)))
    return (
        torch.from_numpy(np.stack(cands, axis=0)).float(),
        np.asarray(mses, dtype=np.float64),
        a_gt_np,
    )


def main(args: Args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = _resolve_paths(args)
    for p in paths:
        if not os.path.isdir(p):
            raise FileNotFoundError(p)

    wm, crit, wm_meta = _load_wm(args.wm_ckpt, device)
    action_dim = int(wm_meta["action_dim"])
    print(f"[offline] wm action_dim={action_dim} meta={wm_meta}", flush=True)

    model_config = AutoConfig.from_pretrained(args.policy_path)
    BRIDGE_TOKENS.update(num_bridge_tokens=model_config.bridge_cfg.get("num_bridge_tokens", 64))
    data_config = load_data_config(
        args.data_config,
        vlm_path=model_config.backbone_cfg["vlm_path"],
        use_bridge=model_config.bridge_cfg.get("use_bridge", True),
        ignore_lang_prefix=True,
    )
    data_config.use_state_future = True
    data_config.state_future_horizon = len(data_config.action_indices)
    data_config.state_future_keys = resolve_state_future_keys(30)
    os.environ["SF_FORCE_KEY_DIM"] = "30"

    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    tokenizer = modality_transform.transforms[-1].vlm_processor.tokenizer
    wm_transforms = _build_wm_transforms(data_config)

    policy = DialPolicy(
        model_path=args.policy_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        tokenizer_len=len(tokenizer),
        denoising_steps=args.denoising_steps,
        device=str(device),
        compute_bridge_loss=False,
    )
    policy.model.eval()
    for p in policy.model.parameters():
        p.requires_grad_(False)

    # Raw loader (policy applies its own transforms).
    results: Dict = {
        "policy_path": args.policy_path,
        "wm_ckpt": args.wm_ckpt,
        "ks": list(args.ks),
        "protocol": {
            "score": "-D(s_hat, s_star) on frames t+1..t+8",
            "D": "group-normalized MSE pos+rot6d",
            "action_mse": "normalized action space vs GT chunk",
            "note": "offline only; online K-sweep is step (5) after this passes",
        },
        "per_k": {},
    }

    for K in args.ks:
        all_delta = []
        all_pair = []
        all_rho = []
        all_top1 = []
        all_mean = []
        n_points = 0

        for path in paths:
            task = os.path.basename(path)
            raw_ds = LeRobotSingleDatasetWithGoalImage(
                dataset_path=path,
                modality_configs=modality_config,
                transforms=None,
                embodiment_tag=args.embodiment_tag,
                split=args.data_split,
                video_backend=args.video_backend,
            )
            # Attach metadata so WM transforms can normalize.
            if hasattr(wm_transforms, "set_metadata"):
                # Use a tiny dataset with transforms to obtain metadata, or set from raw.
                meta_ds = LeRobotSingleDatasetWithGoalImage(
                    dataset_path=path,
                    modality_configs={
                        k: v
                        for k, v in modality_config.items()
                        if k in ("state", "action", "state_future")
                    },
                    transforms=wm_transforms,
                    embodiment_tag=args.embodiment_tag,
                    split=args.data_split,
                    video_backend=args.video_backend,
                )
                _ = meta_ds  # set_metadata happens in dataset init

            n_traj = min(args.trajs, len(raw_ds.trajectory_ids))
            for ti in range(n_traj):
                traj_id = raw_ds.trajectory_ids[ti]
                traj_len = int(raw_ds.trajectory_lengths[ti])
                steps = list(
                    range(0, min(args.points * args.action_horizon, traj_len), args.action_horizon)
                )
                for si, step in enumerate(steps):
                    seed_base = args.seed + zlib.crc32(f"{task}|{ti}|{si}|K{K}".encode()) % 10_000_007
                    obs = raw_ds.get_step_data(traj_id, step)
                    # WM labels in normalized space
                    wm_in = {k: v for k, v in obs.items()}
                    packed = wm_transforms(wm_in)
                    s_t = torch.as_tensor(packed["s_t"], dtype=torch.float32, device=device)
                    s_star = torch.as_tensor(packed["s_future"], dtype=torch.float32, device=device)
                    s_mask = torch.as_tensor(packed["s_mask"], dtype=torch.bool, device=device)

                    a_cands, mses, _ = _sample_candidates(
                        policy, obs, K, action_dim, seed_base
                    )
                    a_cands = a_cands.to(device)
                    s_t_b = s_t.unsqueeze(0).expand(K, -1)
                    s_star_b = s_star.unsqueeze(0).expand(K, -1, -1)
                    s_mask_b = s_mask.unsqueeze(0).expand(K, -1)
                    s_hat = wm(s_t_b, a_cands)
                    D = crit.distance(s_hat, s_star_b, s_mask_b).detach().cpu().numpy()
                    scores = -D

                    top1 = int(np.argmax(scores))
                    top1_mse = float(mses[top1])
                    mean_mse = float(np.mean(mses))
                    delta = mean_mse - top1_mse
                    pair = _pairwise_accuracy(scores, mses)
                    rho = _spearman_score_neg_mse(scores, mses)

                    all_delta.append(delta)
                    all_top1.append(top1_mse)
                    all_mean.append(mean_mse)
                    if not np.isnan(pair):
                        all_pair.append(pair)
                    if not np.isnan(rho):
                        all_rho.append(rho)
                    n_points += 1

            print(
                f"[offline] K={K} finished task={task} "
                f"running_delta={np.mean(all_delta):.5f} n={n_points}",
                flush=True,
            )

        results["per_k"][str(K)] = {
            "n_points": n_points,
            "action_mse_top1": float(np.mean(all_top1)),
            "action_mse_cand_mean": float(np.mean(all_mean)),
            "delta_action_mse_mean_minus_top1": float(np.mean(all_delta)),
            "pairwise_accuracy": float(np.mean(all_pair)) if all_pair else None,
            "spearman_score_vs_neg_mse": float(np.mean(all_rho)) if all_rho else None,
        }
        r = results["per_k"][str(K)]
        print(
            f"[offline] === K={K} === "
            f"Δmse={r['delta_action_mse_mean_minus_top1']:.5f} "
            f"pair_acc={r['pairwise_accuracy']} "
            f"spearman={r['spearman_score_vs_neg_mse']} "
            f"top1_mse={r['action_mse_top1']:.5f} "
            f"mean_mse={r['action_mse_cand_mean']:.5f}",
            flush=True,
        )

    out = args.save_json or str(
        Path(args.wm_ckpt).resolve().parent / "offline_score_validity.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"[offline] wrote {out}", flush=True)
    print(
        "[offline] Gate for (5) online K-sweep: prefer Δmse>0, pairwise>0.5, spearman>0. "
        "Do not start online until this looks OK.",
        flush=True,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
