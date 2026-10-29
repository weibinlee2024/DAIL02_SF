#!/usr/bin/env python3
"""Teacher protocol: shared-pool score vs random + same-state K-branch oracle.

Focus (advisor):
  - Fixed episode seeds
  - Same candidate noise schedule; score/random share the pool; random index
    independent of scores
  - Per-episode config + results
  - At each decision: save/restore, branch-execute all K candidates
  - Check: dynamics (ŝ) vs real wrist outcome, SF target (s*) vs task progress,
    score ranking vs oracle ranking, grasp/transport/placement/close first-fail,
    smoothness / mean-candidate shortcuts
  - Evidence for: why offline pairwise ~ random / Spearman ~ 0, yet online
    score−random reached +21.5pp on aligned attr4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gr00t.experiment.data_config_dial import load_data_config  # noqa: E402
from gr00t.eval.simulation_augPosRot import (  # noqa: E402
    MultiStepConfig,
    SimulationConfig,
    VideoConfig,
    _create_single_env,
)
from gr00t.model.transforms import BRIDGE_TOKENS  # noqa: E402
from gr00t.model.policy_dial import DialPolicy  # noqa: E402


STAGES = ("grasp", "transport", "placement", "close")


def _unwrap_to_sim_holder(env):
    e = env
    seen = set()
    while id(e) not in seen:
        seen.add(id(e))
        if hasattr(e, "sim") and hasattr(getattr(e, "sim", None), "get_state"):
            return e
        if hasattr(e, "env"):
            e = e.env
            continue
        uw = getattr(e, "unwrapped", None)
        if uw is not None and uw is not e:
            e = uw
            continue
        break
    raise RuntimeError("could not find robosuite env with .sim")


def _save_state(sim_env) -> np.ndarray:
    return np.array(sim_env.sim.get_state().flatten())


def _restore_state(sim_env, state: np.ndarray) -> None:
    sim_env.sim.set_state_from_flattened(state)
    sim_env.sim.forward()


def _task_host(sim_env):
    if getattr(sim_env, "microwave", None) is not None:
        return sim_env
    objects = getattr(sim_env, "objects", None)
    if isinstance(objects, dict) and "obj" in objects:
        return sim_env
    if hasattr(sim_env, "env"):
        return sim_env.env
    return sim_env


def _microwave_snapshot(sim_env) -> Dict[str, Any]:
    """Door / grasp / inside / success + stage flags for Close microwave PnP."""
    out: Dict[str, Any] = {
        "door": None,
        "grasped": None,
        "inside": None,
        "success": None,
        "stages": {s: False for s in STAGES},
    }
    host = _task_host(sim_env)
    try:
        mw = getattr(host, "microwave", None)
        if mw is not None:
            out["door"] = float(mw.get_door_state(env=host)["door"])
        if hasattr(host, "_check_success"):
            out["success"] = bool(host._check_success())
        # grasp
        try:
            grasped = bool(
                host._check_grasp(
                    gripper=host.robots[0].gripper["right"],
                    object_geoms=host.objects["obj"],
                )
            )
            out["grasped"] = grasped
        except Exception:
            out["grasped"] = None
        # inside
        try:
            from robocasa.utils.object_utils import obj_inside_of

            out["inside"] = bool(
                obj_inside_of(
                    env=host,
                    obj_name=host.objects["obj"].name,
                    fixture_id=host.microwave,
                    partial_check=True,
                )
            )
        except Exception:
            try:
                import robocasa.utils.object_utils as OU

                out["inside"] = bool(OU.check_obj_fixture_contact(host, "obj", host.microwave))
            except Exception:
                out["inside"] = None

        door = out["door"]
        grasped = bool(out["grasped"])
        inside = bool(out["inside"])
        success = bool(out["success"])
        # Close task: door usually starts open; progress = grasp → place → close.
        out["stages"]["grasp"] = grasped or inside or success
        out["stages"]["transport"] = (grasped and door is not None and door >= 0.05) or inside or success
        out["stages"]["placement"] = inside or success
        out["stages"]["close"] = success or (
            inside and door is not None and door <= 0.005
        )
        out["family"] = "microwave"
    except Exception as exc:  # pragma: no cover
        out["error"] = str(exc)
    return out


def _place_snapshot(sim_env) -> Dict[str, Any]:
    """Grasp / lifted-off-counter / in-container for counter-to-container PnP.

    There is no door, so ``close`` is not a stage. Transport means the object
    is grasped and at least 3 cm above the height recorded while it was still
    resting. Placement uses the same receptacle check as ``_check_success``.
    """
    out: Dict[str, Any] = {
        "family": "place",
        "grasped": None,
        "lifted": None,
        "inside": None,
        "success": None,
        "obj_z": None,
        "stages": {"grasp": False, "transport": False, "placement": False},
    }
    host = _task_host(sim_env)
    try:
        import robocasa.utils.object_utils as OU

        if hasattr(host, "_check_success"):
            out["success"] = bool(host._check_success())
        try:
            grasped = bool(
                host._check_grasp(
                    gripper=host.robots[0].gripper["right"],
                    object_geoms=host.objects["obj"],
                )
            )
        except Exception:
            grasped = False
        out["grasped"] = grasped
        try:
            inside = bool(OU.check_obj_in_receptacle(host, "obj", "container"))
        except Exception:
            inside = False
        out["inside"] = inside
        obj = host.objects["obj"]
        name = getattr(obj, "name", "obj")
        body_ids = host.obj_body_id
        bid = body_ids[name] if name in body_ids else body_ids["obj"]
        z = float(host.sim.data.body_xpos[bid][2])
        out["obj_z"] = z
        if not grasped and not inside:
            host._probe_obj_rest_z = z
        rest = float(getattr(host, "_probe_obj_rest_z", z))
        lifted = grasped and (z - rest) > 0.03
        out["lifted"] = lifted
        success = bool(out["success"])
        out["stages"]["grasp"] = grasped or inside or success
        out["stages"]["transport"] = lifted or inside or success
        out["stages"]["placement"] = inside or success
    except Exception as exc:  # pragma: no cover
        out["error"] = str(exc)
    return out


def _task_snapshot(sim_env) -> Dict[str, Any]:
    host = _task_host(sim_env)
    if getattr(host, "microwave", None) is not None:
        return _microwave_snapshot(sim_env)
    return _place_snapshot(sim_env)


def _first_fail_stage(stage_hist: List[Dict[str, bool]], success: bool) -> Optional[str]:
    """Earliest recorded stage that never becomes true (if not success)."""
    if success:
        return None
    present = set()
    for snap in stage_hist:
        present.update(snap.keys())
    achieved = {s: False for s in present}
    for snap in stage_hist:
        for s in present:
            achieved[s] = achieved[s] or bool(snap.get(s))
    for s in ("grasp", "transport", "placement", "close"):
        if s in present and not achieved[s]:
            return s
    return "unknown"


def _as_time_major_action(action_dict: Dict[str, Any], n_action_steps: int) -> Dict[str, np.ndarray]:
    """Normalize unnorm action dict to {key: [T, D]} with T == n_action_steps.

    DialPolicy may leave a leading batch dim when obs were batched → (1, H, D).
    Policy horizon H is often 16 while eval commits only n_action_steps (8).
    """
    out: Dict[str, np.ndarray] = {}
    for key, value in action_dict.items():
        arr = np.asarray(value)
        # Drop leading batch dims of size 1: (1, H, D) → (H, D)
        while arr.ndim >= 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            # Single-step vector — repeat not expected; keep as (1, D)
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"action {key} has unexpected shape {arr.shape}")
        out[key] = arr[:n_action_steps]
    return out


def _step_chunk_on_base(base_gym, action_dict: Dict[str, Any], n_action_steps: int) -> None:
    """Execute first n_action_steps of a policy chunk on the unwrapped gym env."""
    chunk = _as_time_major_action(action_dict, n_action_steps)
    for t in range(n_action_steps):
        act = {key: arr[t] for key, arr in chunk.items()}
        base_gym.step(act)


def _obs_for_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in obs.items():
        if isinstance(v, str):
            out[k] = (v,)
        else:
            out[k] = v[None, :] if isinstance(v, np.ndarray) else np.array(v)[None, ...]
    if "video.ego_view_bg_crop_pad_res256_freq20" in out and "video.ego_view" not in out:
        out["video.ego_view"] = out.pop("video.ego_view_bg_crop_pad_res256_freq20")
    return out


def _extract_wrist18_from_obs(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    """Best-effort raw wrist pos+rot6d from unnormalized obs keys (not model space)."""
    keys = [
        "state.wrist_r_pos",
        "state.wrist_r_rot6d",
        "state.wrist_l_pos",
        "state.wrist_l_rot6d",
    ]
    parts = []
    for k in keys:
        if k not in obs:
            return None
        v = np.asarray(obs[k], dtype=np.float32)
        if v.ndim == 2:
            v = v[-1]
        parts.append(v.reshape(-1))
    return np.concatenate(parts, axis=0)


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 or dy == 0:
                continue
            if dx == dy:
                concord += 1
            else:
                discord += 1
    denom = concord + discord
    return float((concord - discord) / denom) if denom else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def _pairwise_agree(score: np.ndarray, oracle: np.ndarray) -> Dict[str, float]:
    """Offline-style pairwise: among pairs with oracle preference, does score agree?"""
    n = len(score)
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if oracle[i] == oracle[j]:
                continue
            total += 1
            prefer_i = oracle[i] > oracle[j]
            score_i = score[i] > score[j]
            if prefer_i == score_i:
                agree += 1
    return {
        "pairwise_n": float(total),
        "pairwise_acc": float(agree / total) if total else float("nan"),
    }


def _oracle_proxy(pre: Dict[str, Any], post: Dict[str, Any]) -> float:
    """Higher = better real chunk outcome. Place tasks have no door."""
    if pre.get("family") == "place" or post.get("family") == "place":
        g0, g1 = bool(pre.get("grasped")), bool(post.get("grasped"))
        l0, l1 = bool(pre.get("lifted")), bool(post.get("lifted"))
        in0, in1 = bool(pre.get("inside")), bool(post.get("inside"))
        suc1 = 1.0 if post.get("success") else 0.0
        gain = 0.0
        if (not g0) and g1:
            gain += 1.0
        if (not l0) and l1:
            gain += 1.0
        if (not in0) and in1:
            gain += 3.0
        if g1:
            gain += 0.2
        gain += 10.0 * suc1
        return float(gain)
    door0, door1 = pre.get("door"), post.get("door")
    g0, g1 = bool(pre.get("grasped")), bool(post.get("grasped"))
    in0, in1 = bool(pre.get("inside")), bool(post.get("inside"))
    suc1 = 1.0 if post.get("success") else 0.0
    # Stage gains
    gain = 0.0
    if (not g0) and g1:
        gain += 1.0
    if (not in0) and in1:
        gain += 3.0
    if door0 is not None and door1 is not None:
        if in1 or in0:
            gain += float(max(0.0, door0 - door1)) * 2.0  # closing
        else:
            # keep door open while approaching / grasping
            gain += 0.1 * float(door1)
    gain += 10.0 * suc1
    # small bonus for maintaining grasp
    if g1:
        gain += 0.2
    return float(gain)


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _find_base_gym(env):
    base_gym = env
    while hasattr(base_gym, "env"):
        nxt = base_gym.env
        if type(nxt).__name__ == "GrootRoboCasaEnv" or hasattr(nxt, "key_converter"):
            return nxt
        base_gym = nxt
    return base_gym


def _run_pass(
    *,
    args: argparse.Namespace,
    policy: DialPolicy,
    env,
    base_gym,
    sim_env,
    episode_seeds: List[int],
    commit_mode: str,
    saved_pools: Optional[List[List[Dict[str, Any]]]] = None,
    record_pools: bool = False,
    do_branch: bool = True,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """One full pass over episodes.

    commit_mode: 'score' | 'random'
    If saved_pools is provided, reuse candidate pools (exact shared set).
    If record_pools, store per-decision meta needed to replay.
    """
    assert commit_mode in ("score", "random")
    all_ep_records: List[Dict[str, Any]] = []
    pools_out: List[List[Dict[str, Any]]] = []

    for ep, ep_seed in enumerate(episode_seeds):
        obs, _ = env.reset(seed=int(ep_seed))
        host = _task_host(sim_env)
        if hasattr(host, "_probe_obj_rest_z"):
            delattr(host, "_probe_obj_rest_z")
        done = False
        step_i = 0
        ep_success = False
        stage_hist: List[Dict[str, bool]] = []
        decisions: List[Dict[str, Any]] = []
        ep_pools: List[Dict[str, Any]] = []

        while not done and step_i * args.n_action_steps < args.max_episode_steps:
            state = _save_state(sim_env)
            pre = _task_snapshot(sim_env)
            stage_hist.append(dict(pre["stages"]))

            if saved_pools is not None:
                if step_i >= len(saved_pools[ep]):
                    break
                meta = dict(saved_pools[ep][step_i]["meta"])
                # JSON round-trip → restore numpy action dicts for env.step
                restored_unnorm = []
                for ad in meta["actions_unnorm"]:
                    restored_unnorm.append({kk: np.asarray(vv) for kk, vv in ad.items()})
                meta["actions_unnorm"] = restored_unnorm
                meta["dist_to_mean"] = np.asarray(meta["dist_to_mean"], dtype=np.float64)
                meta["smoothness"] = np.asarray(meta["smoothness"], dtype=np.float64)
                meta["scores_pos"] = np.asarray(meta["scores_pos"], dtype=np.float64)
                meta["scores_rot"] = np.asarray(meta["scores_rot"], dtype=np.float64)
            else:
                meta = policy.propose_bon_with_meta(_obs_for_policy(obs))
                if meta.get("scores") is None:
                    raise RuntimeError("WM scores missing; check wm_ckpt")

            scores = np.asarray(meta["scores"], dtype=np.float64)
            pick_score = int(meta["best"])
            pick_random = int(meta["random_pick"])
            commit = pick_score if commit_mode == "score" else pick_random

            branch_rows = []
            if do_branch and commit_mode == "score" and saved_pools is None:
                # Branch all K from the same state (oracle / dynamics checks).
                s_star = meta.get("pred_sf")
                s_star_arr = np.asarray(s_star, dtype=np.float32) if s_star is not None else None
                for k in range(args.bon_k):
                    _restore_state(sim_env, state)
                    _step_chunk_on_base(base_gym, meta["actions_unnorm"][k], args.n_action_steps)
                    post = _task_snapshot(sim_env)
                    # dynamics: D(ŝ, s*) already = -score; also report ||mean ŝ_{1:9}|| and
                    # ||mean ŝ - mean s*|| as SF-alignment diagnostics.
                    dyn = {}
                    s_hat = meta.get("s_hat")
                    if s_hat is not None:
                        sh = np.asarray(s_hat[k], dtype=np.float32)
                        dyn["s_hat_window_mean_norm"] = float(np.linalg.norm(sh[1:9].mean(axis=0)))
                        if s_star_arr is not None and s_star_arr.shape == sh.shape:
                            dyn["s_hat_minus_s_star_window_mse"] = float(
                                ((sh[1:9] - s_star_arr[1:9]) ** 2).mean()
                            )
                    oracle = _oracle_proxy(pre, post)
                    branch_rows.append(
                        {
                            "k": k,
                            "score": float(scores[k]),
                            "score_pos": float(meta["scores_pos"][k]),
                            "score_rot": float(meta["scores_rot"][k]),
                            "dist_to_mean": float(meta["dist_to_mean"][k]),
                            "smoothness": float(meta["smoothness"][k]),
                            "oracle_proxy": oracle,
                            "post": _to_jsonable(post),
                            "dynamics": dyn,
                        }
                    )
                _restore_state(sim_env, state)

            # Commit chosen candidate via MultiStepWrapper (keeps obs buffers consistent).
            chosen = _as_time_major_action(meta["actions_unnorm"][commit], args.n_action_steps)
            obs, reward, terminations, truncations, infos = env.step(chosen)
            ep_success |= bool(np.asarray(infos.get("success", False)).reshape(-1)[0])
            done = bool(
                np.asarray(terminations).reshape(-1)[0] or np.asarray(truncations).reshape(-1)[0]
            )
            post_commit = _task_snapshot(sim_env)
            stage_hist.append(dict(post_commit["stages"]))

            dec: Dict[str, Any] = {
                "decision": step_i,
                "episode_seed": int(ep_seed),
                "seed_base": int(meta["seed_base"]),
                "pick_score": pick_score,
                "pick_random": pick_random,
                "commit_mode": commit_mode,
                "commit": commit,
                "cand_dispersion": float(meta.get("cand_dispersion", 0.0)),
                "pre": _to_jsonable(pre),
                "post_commit": _to_jsonable(post_commit),
                "scores": scores.tolist(),
                "scores_pos": np.asarray(meta["scores_pos"]).tolist(),
                "scores_rot": np.asarray(meta["scores_rot"]).tolist(),
                "dist_to_mean": np.asarray(meta["dist_to_mean"]).tolist(),
                "smoothness": np.asarray(meta["smoothness"]).tolist(),
                "s_t": _to_jsonable(meta.get("s_t")),
                "pred_sf": _to_jsonable(meta.get("pred_sf")),
                # s_hat can be large; store for score pass only
                "s_hat": _to_jsonable(meta.get("s_hat")) if do_branch and commit_mode == "score" else None,
            }

            if branch_rows:
                score_arr = np.array([r["score"] for r in branch_rows], dtype=np.float64)
                oracle_arr = np.array([r["oracle_proxy"] for r in branch_rows], dtype=np.float64)
                pw = _pairwise_agree(score_arr, oracle_arr)
                dec.update(
                    {
                        "branches": branch_rows,
                        "kendall_score_vs_oracle": _kendall_tau(score_arr, oracle_arr),
                        "spearman_score_vs_oracle": _spearman(score_arr, oracle_arr),
                        "pairwise_acc_score_vs_oracle": pw["pairwise_acc"],
                        "pairwise_n": pw["pairwise_n"],
                        "score_pick_is_oracle_best": bool(pick_score == int(np.argmax(oracle_arr))),
                        "random_pick_is_oracle_best": bool(pick_random == int(np.argmax(oracle_arr))),
                        "corr_score_vs_dist_to_mean": float(
                            np.corrcoef(score_arr, meta["dist_to_mean"])[0, 1]
                        )
                        if np.std(score_arr) > 1e-8 and np.std(meta["dist_to_mean"]) > 1e-8
                        else float("nan"),
                        "corr_score_vs_smoothness": float(
                            np.corrcoef(score_arr, meta["smoothness"])[0, 1]
                        )
                        if np.std(score_arr) > 1e-8 and np.std(meta["smoothness"]) > 1e-8
                        else float("nan"),
                        # SF target vs task: does higher score align with oracle progress?
                        "sf_vs_progress_spearman": _spearman(score_arr, oracle_arr),
                    }
                )

            decisions.append(dec)

            if record_pools:
                # Persist everything needed to replay the exact same candidate set.
                ep_pools.append(
                    {
                        "meta": {
                            "best": pick_score,
                            "random_pick": pick_random,
                            "scores": scores.tolist(),
                            "scores_pos": np.asarray(meta["scores_pos"]).tolist(),
                            "scores_rot": np.asarray(meta["scores_rot"]).tolist(),
                            "dist_to_mean": np.asarray(meta["dist_to_mean"]).tolist(),
                            "smoothness": np.asarray(meta["smoothness"]).tolist(),
                            "cand_dispersion": float(meta.get("cand_dispersion", 0.0)),
                            "actions_norm": _to_jsonable(meta["actions_norm"]),
                            "actions_unnorm": _to_jsonable(meta["actions_unnorm"]),
                            "seed_base": int(meta["seed_base"]),
                            "s_t": _to_jsonable(meta.get("s_t")),
                            "s_hat": _to_jsonable(meta.get("s_hat")),
                            "pred_sf": _to_jsonable(meta.get("pred_sf")),
                        }
                    }
                )

            step_i += 1

        first_fail = _first_fail_stage(stage_hist, ep_success)
        all_ep_records.append(
            {
                "episode": ep,
                "episode_seed": int(ep_seed),
                "commit_mode": commit_mode,
                "success": bool(ep_success),
                "n_decisions": len(decisions),
                "first_fail_stage": first_fail,
                "decisions": decisions,
            }
        )
        if record_pools:
            pools_out.append(ep_pools)

    return all_ep_records, pools_out


def run_probe(args: argparse.Namespace) -> Path:
    model_config = AutoConfig.from_pretrained(args.model_path)
    num_bridge_tokens = model_config.bridge_cfg.get("num_bridge_tokens", 64)
    BRIDGE_TOKENS.update(num_bridge_tokens=num_bridge_tokens)
    ignore_lang_prefix = getattr(model_config, "ignore_lang_prefix", False)
    data_config = load_data_config(
        args.data_config,
        vlm_path=model_config.backbone_cfg["vlm_path"],
        use_bridge=model_config.bridge_cfg["use_bridge"],
        ignore_lang_prefix=ignore_lang_prefix,
    )
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    tokenizer = modality_transform.transforms[-1].vlm_processor.tokenizer

    policy = DialPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        tokenizer_len=len(tokenizer),
        sf_eval_mode=args.sf_eval_mode,
        sf_swap_bank_max=args.sf_swap_bank_max,
        wm_ckpt=args.wm_ckpt,
        bon_k=args.bon_k,
        bon_seed=args.bon_seed,
        bon_select="score",
    )

    # Must match online eval client: inject wrist pos/rot6d via VideoRecordingWrapper.
    state_keys = set(modality_config["state"].modality_keys)
    cfg = SimulationConfig(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        n_envs=1,
        video=VideoConfig(video_dir=None),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
        ),
        state_modality_keys=state_keys,
    )
    env = _create_single_env(cfg, idx=0)
    base_gym = _find_base_gym(env)
    sim_env = _unwrap_to_sim_holder(base_gym)
    print(f"[probe] base_gym={type(base_gym).__name__} sim_holder={type(sim_env).__name__}", flush=True)

    episode_seeds = [int(args.seed_base) + i for i in range(args.n_episodes)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    config_blob = {
        "env_name": args.env_name,
        "model_path": args.model_path,
        "wm_ckpt": args.wm_ckpt,
        "bon_k": args.bon_k,
        "bon_seed": args.bon_seed,
        "n_action_steps": args.n_action_steps,
        "max_episode_steps": args.max_episode_steps,
        "n_episodes": args.n_episodes,
        "seed_base": args.seed_base,
        "episode_seeds": episode_seeds,
        "protocol": (
            "Pass1 score: sample pool, branch all K, commit score pick. "
            "Pass2 random: same episode seeds + exact saved pools, commit random_pick. "
            "random_pick = RandomState(seed_base+17) independent of scores."
        ),
    }

    print("[probe] pass1=score (branch oracle + save pools)", flush=True)
    score_eps, pools = _run_pass(
        args=args,
        policy=policy,
        env=env,
        base_gym=base_gym,
        sim_env=sim_env,
        episode_seeds=episode_seeds,
        commit_mode="score",
        saved_pools=None,
        record_pools=True,
        do_branch=True,
    )

    print("[probe] pass2=random (replay saved pools, no branch)", flush=True)
    # Reset policy bon call counter so logs stay sane; pools are reused so sampling not needed.
    if hasattr(policy, "_wm_bon") and policy._wm_bon is not None:
        policy._wm_bon._call_idx = 0
    random_eps, _ = _run_pass(
        args=args,
        policy=policy,
        env=env,
        base_gym=base_gym,
        sim_env=sim_env,
        episode_seeds=episode_seeds,
        commit_mode="random",
        saved_pools=pools,
        record_pools=False,
        do_branch=False,
    )

    env.close()

    # ---- aggregates aimed at offline-vs-online gap ----
    def _sr(eps):
        return float(np.mean([e["success"] for e in eps])) if eps else float("nan")

    decs = [d for e in score_eps for d in e["decisions"] if "branches" in d]
    pairwise_accs = [d["pairwise_acc_score_vs_oracle"] for d in decs if d.get("pairwise_n", 0) > 0]
    spearmans = [d["spearman_score_vs_oracle"] for d in decs]
    kendalls = [d["kendall_score_vs_oracle"] for d in decs]
    shortcut_mean = [d["corr_score_vs_dist_to_mean"] for d in decs]
    shortcut_smooth = [d["corr_score_vs_smoothness"] for d in decs]
    fail_score = [e["first_fail_stage"] for e in score_eps]
    fail_rand = [e["first_fail_stage"] for e in random_eps]

    def _count_stages(fails):
        out = {s: 0 for s in STAGES}
        out["none_success"] = 0
        out["unknown"] = 0
        for f in fails:
            if f is None:
                out["none_success"] += 1
            elif f in out:
                out[f] += 1
            else:
                out["unknown"] += 1
        return out

    summary = {
        "env_name": args.env_name,
        "n_episodes": args.n_episodes,
        "elapsed_s": time.time() - t0,
        "sr_score": _sr(score_eps),
        "sr_random": _sr(random_eps),
        "sr_delta_score_minus_random": _sr(score_eps) - _sr(random_eps),
        "n_branch_decisions": len(decs),
        "mean_pairwise_acc_score_vs_oracle": float(np.nanmean(pairwise_accs)) if pairwise_accs else None,
        "mean_spearman_score_vs_oracle": float(np.nanmean(spearmans)) if spearmans else None,
        "mean_kendall_score_vs_oracle": float(np.nanmean(kendalls)) if kendalls else None,
        "frac_score_pick_is_oracle_best": float(
            np.mean([d["score_pick_is_oracle_best"] for d in decs])
        )
        if decs
        else None,
        "frac_random_pick_is_oracle_best": float(
            np.mean([d["random_pick_is_oracle_best"] for d in decs])
        )
        if decs
        else None,
        "mean_corr_score_vs_dist_to_mean": float(np.nanmean(shortcut_mean)) if shortcut_mean else None,
        "mean_corr_score_vs_smoothness": float(np.nanmean(shortcut_smooth)) if shortcut_smooth else None,
        "first_fail_stage_counts_score": _count_stages(fail_score),
        "first_fail_stage_counts_random": _count_stages(fail_rand),
        "interpretation_hints": {
            "offline_gap": (
                "If mean_pairwise_acc ~ 0.5 and Spearman ~ 0 on these online pools, "
                "verifier ranking is weak at decision level (matches offline). "
                "If pairwise_acc is clearly > 0.5 here but offline was ~0.5, the gap is "
                "distribution shift (offline holdout vs online on-policy states)."
            ),
            "online_plus21pp": (
                "sr_delta_score_minus_random is the shared-pool, fixed-seed analogue of "
                "aligned attr4 K4-score vs K4-random (+21.5pp). Even with weak Spearman, "
                "top-1 hit-rate (frac_score_pick_is_oracle_best) can still beat random "
                "and compound over the episode."
            ),
            "shortcuts": (
                "corr_score_vs_dist_to_mean > 0 => high score farther from mean (not mean-seeking). "
                "corr_score_vs_smoothness < 0 => high score prefers smaller action deltas (smoother)."
            ),
        },
    }

    payload = {
        "config": config_blob,
        "summary": summary,
        "score_episodes": _to_jsonable(score_eps),
        "random_episodes": _to_jsonable(random_eps),
    }
    out_path = out_dir / f"bon_teacher_probe_{args.tag}.json"
    # pools can be huge; write separately without duplicating actions twice if needed
    out_path.write_text(json.dumps(payload, indent=2))
    pools_path = out_dir / f"bon_teacher_pools_{args.tag}.json"
    pools_path.write_text(json.dumps(_to_jsonable(pools), indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {pools_path}")
    return out_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        default="/media/weibin/Data/mean_pool_motcond_w0p05_tuneLLM_L12_ft_20000/checkpoint-20000",
    )
    p.add_argument(
        "--wm_ckpt",
        default="/media/weibin/weibin01/DIAL_SF_20261018/wm_verifier_posrot18_ft20k_end2end/checkpoint-20000.pt",
    )
    p.add_argument(
        "--data_config",
        default="fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop",
    )
    p.add_argument("--embodiment_tag", default="gr1")
    p.add_argument("--sf_eval_mode", default="correct")
    p.add_argument("--sf_swap_bank_max", type=int, default=64)
    p.add_argument("--denoising_steps", type=int, default=None)
    p.add_argument("--bon_k", type=int, default=4)
    p.add_argument("--bon_seed", type=int, default=0)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--max_episode_steps", type=int, default=720)
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--seed_base", type=int, default=20261028)
    p.add_argument(
        "--env_name",
        default="gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    )
    p.add_argument(
        "--out_dir",
        default="/media/weibin/weibin01/DIAL_SF_20261018/bon_attribution_4arm/branch_probe",
    )
    p.add_argument("--tag", default="potato_mw_k4n8")
    return p


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("SF_FORCE_KEY_DIM", "30")
    os.environ.setdefault("SKIP_VIDEO", "1")
    run_probe(build_argparser().parse_args())
