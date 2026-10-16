#!/usr/bin/env python3
"""Normalized SF probe (teacher checklist, 2026-10).

Requirements:
1) Cross-trajectory train/test split (no traj leakage).
2) Report t=0 vs future frames separately.
3) Group R²: pos / rot6d / hand (for D_sf=60 tip pos folded into pos).
4) Probe inputs: bridge tokens AND SF decoder mid_hidden
   (query_ca: true mid_hidden; mean_pool: mlp first-layer hidden as proxy).
5) Conclusion fields: info present? which layer? which state group?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

TASKS = [
    "PosttrainPnPNovelFromPlacematToPlateSplitA",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "PosttrainPnPNovelFromTrayToPlateSplitA",
]

KEYS_D30 = [
    "state.wrist_r_pos",
    "state.wrist_r_rot6d",
    "state.wrist_l_pos",
    "state.wrist_l_rot6d",
    "state.right_hand",
    "state.left_hand",
]
KEYS_D60 = KEYS_D30 + [
    "state.thumb_r_pos",
    "state.index_r_pos",
    "state.middle_r_pos",
    "state.ring_r_pos",
    "state.pinky_r_pos",
    "state.thumb_l_pos",
    "state.index_l_pos",
    "state.middle_l_pos",
    "state.ring_l_pos",
    "state.pinky_l_pos",
]
KEY_DIMS = {
    "state.wrist_r_pos": 3,
    "state.wrist_r_rot6d": 6,
    "state.wrist_l_pos": 3,
    "state.wrist_l_rot6d": 6,
    "state.right_hand": 6,
    "state.left_hand": 6,
    "state.thumb_r_pos": 3,
    "state.index_r_pos": 3,
    "state.middle_r_pos": 3,
    "state.ring_r_pos": 3,
    "state.pinky_r_pos": 3,
    "state.thumb_l_pos": 3,
    "state.index_l_pos": 3,
    "state.middle_l_pos": 3,
    "state.ring_l_pos": 3,
    "state.pinky_l_pos": 3,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--data-root",
        default="/media/weibin/weibin02/LeRobot-AugPosRot-Correct",
    )
    p.add_argument(
        "--data-config",
        default="fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-train-trajs", type=int, default=48)
    p.add_argument("--num-test-trajs", type=int, default=16)
    p.add_argument("--steps-per-traj", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def group_slices(dim: int):
    """Return {pos, rot6d, hand} index lists for one frame of D_sf."""
    keys = KEYS_D60 if dim >= 60 else KEYS_D30
    if sum(KEY_DIMS[k] for k in keys) != dim and dim not in (30, 60):
        raise ValueError(f"unsupported D_sf={dim}")
    if dim == 30:
        keys = KEYS_D30
    elif dim == 60:
        keys = KEYS_D60
    else:
        # fall back to prefix of KEYS_D60 matching dim
        keys = []
        s = 0
        for k in KEYS_D60:
            if s + KEY_DIMS[k] > dim:
                break
            keys.append(k)
            s += KEY_DIMS[k]
        if s != dim:
            raise ValueError(f"cannot build group slices for dim={dim}")

    groups = {"pos": [], "rot6d": [], "hand": []}
    off = 0
    for k in keys:
        d = KEY_DIMS[k]
        sl = list(range(off, off + d))
        if k.endswith("_pos") or "pos" in k.split(".")[-1]:
            groups["pos"].extend(sl)
        elif "rot6d" in k:
            groups["rot6d"].extend(sl)
        elif "hand" in k:
            groups["hand"].extend(sl)
        else:
            groups["pos"].extend(sl)
        off += d
    return groups


def build_singles(args, model_config, sf_dim: int):
    from gr00t.data.dataset import LeRobotSingleDatasetWithGoalImage
    from gr00t.experiment.data_config_dial import load_data_config
    from gr00t.model.transforms import BRIDGE_TOKENS

    num_bridge = model_config.bridge_cfg.get("num_bridge_tokens", 64)
    BRIDGE_TOKENS.update(num_bridge_tokens=num_bridge)

    data_config_cls = load_data_config(
        args.data_config,
        vlm_path=model_config.backbone_cfg["vlm_path"],
        use_bridge=True,
        ignore_lang_prefix=True,
    )
    data_config_cls.use_state_future = True
    data_config_cls.state_future_horizon = len(data_config_cls.action_indices)
    # Match checkpoint SF dim (critical when workspace default keys are 60).
    data_config_cls.state_future_keys = list(KEYS_D60 if sf_dim >= 60 else KEYS_D30)

    singles = []
    for t in TASKS:
        path = Path(args.data_root) / f"gr1_unified.{t}"
        if not path.is_dir():
            raise FileNotFoundError(path)
        ds = LeRobotSingleDatasetWithGoalImage(
            dataset_path=str(path),
            modality_configs=data_config_cls.modality_config(),
            transforms=data_config_cls.transform(),
            embodiment_tag="gr1",
            split="[:-10]",
            video_backend="decord",
        )
        singles.append((t, ds))
    return singles, data_config_cls


def load_model(args):
    from transformers import AutoConfig
    from gr00t.model.gr00t_n1_dial import GR00T_N1_5_DIAL
    from gr00t.model.transforms import DefaultDataCollator, BRIDGE_TOKENS

    model_config = AutoConfig.from_pretrained(args.model_path)
    num_bridge = model_config.bridge_cfg.get("num_bridge_tokens", 64)
    BRIDGE_TOKENS.update(num_bridge_tokens=num_bridge)
    tokenizer_len = model_config.bridge_cfg.get("tokenizer_len") or 151729
    sf_dim = int(model_config.bridge_cfg.get("state_future_dim") or 30)
    decoder = model_config.bridge_cfg.get("state_future_decoder", "query_ca")
    model = GR00T_N1_5_DIAL.from_pretrained(
        args.model_path,
        tokenizer_len=int(tokenizer_len),
        use_state_future=True,
        compute_state_future_loss=True,
        state_future_loss_weight=0.5,
        state_future_decoder=decoder,
        state_future_dim=sf_dim,
        compute_bridge_loss=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    collator = DefaultDataCollator(vlm_path=model.config.backbone_cfg["vlm_path"])
    return model, model_config, collator, sf_dim, decoder


def sample_traj_steps(singles, num_train_trajs, num_test_trajs, steps_per_traj, seed):
    """Return train/test lists of (task, ds, traj_id, base_index)."""
    rng = np.random.RandomState(seed)
    # pool: (task_i, traj_id, list of base_index)
    pool = []
    for task_i, (task, ds) in enumerate(singles):
        # all_steps: list[(traj_id, base_index)]
        by_traj = {}
        for traj_id, base_index in ds.all_steps:
            by_traj.setdefault(int(traj_id), []).append(int(base_index))
        for traj_id, bases in by_traj.items():
            pool.append((task_i, task, traj_id, bases))

    rng.shuffle(pool)
    n_test = min(num_test_trajs, max(1, int(round(len(pool) * 0.25))))
    # honor explicit counts when pool is large enough
    if len(pool) >= num_train_trajs + num_test_trajs:
        n_test = num_test_trajs
        n_train = num_train_trajs
    else:
        n_train = len(pool) - n_test
    test_pool = pool[:n_test]
    train_pool = pool[n_test : n_test + n_train]

    def expand(traj_pool, tag):
        out = []
        for task_i, task, traj_id, bases in traj_pool:
            if not bases:
                continue
            pick = rng.choice(bases, size=min(steps_per_traj, len(bases)), replace=False)
            for b in pick:
                out.append(
                    {
                        "split": tag,
                        "task": task,
                        "task_i": task_i,
                        "traj_id": int(traj_id),
                        "base_index": int(b),
                    }
                )
        return out

    return expand(train_pool, "train"), expand(test_pool, "test"), {
        "n_train_trajs": len(train_pool),
        "n_test_trajs": len(test_pool),
        "n_pool_trajs": len(pool),
    }


@torch.no_grad()
def extract_one(model, sample, device, decoder: str):
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    # collator expects list of samples; caller batches
    raise RuntimeError("use extract_batch")


@torch.no_grad()
def extract_batch(model, batch, device, decoder: str):
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    backbone_inputs, action_inputs = model.prepare_input(batch)
    backbone_outputs = model.backbone(backbone_inputs)
    feat = backbone_outputs["backbone_features"][:, -model.num_bridge_tokens :]
    B = feat.shape[0]
    mid = None
    mlp_h = None
    if decoder == "query_ca":
        pred, mid = model._predict_state_future(feat, B, return_hidden=True)
    else:
        pred = model._predict_state_future(feat, B, return_hidden=False)
        pooled = feat.mean(dim=1)
        # Sequential(Linear, GELU, Linear)
        h = model.state_future_projector[0](pooled)
        mlp_h = model.state_future_projector[1](h)
    gt = action_inputs.get("state_future")
    mask = action_inputs.get("state_future_mask")
    if gt is None or pred is None:
        raise RuntimeError("missing state_future GT or decoder")
    gt = gt.to(pred.dtype)
    if mask is not None:
        mask = mask.to(pred.dtype)
    return feat, pred, gt, mask, mid, mlp_h


def per_frame_mse(preds, gts, masks):
    pred = torch.cat(preds, dim=0)
    gt = torch.cat(gts, dim=0)
    mask = torch.ones_like(pred) if masks[0] is None else torch.cat(masks, dim=0)
    diff = (pred - gt) ** 2
    T = pred.shape[1]
    per = []
    for t in range(T):
        m = mask[:, t]
        per.append(float((diff[:, t] * m).sum() / m.sum().clamp_min(1.0)))
    return {
        "per_frame_mse": per,
        "frame0": per[0],
        "future_1_to_H": float(np.mean(per[1:])) if len(per) > 1 else None,
        "near_1_4": float(np.mean(per[1:5])) if len(per) > 4 else None,
        "far_last4": float(np.mean(per[-4:])) if len(per) >= 4 else None,
        "all_frames": float(np.mean(per)),
        "ratio_future_over_frame0": (
            float(np.mean(per[1:]) / per[0]) if per[0] > 1e-12 else None
        ),
    }


def _r2(y_true, y_pred):
    if y_true.size == 0:
        return None
    return float(r2_score(y_true, y_pred, multioutput="uniform_average"))


def fit_probe(Xtr, Xte, Ytr, Yte, dim, groups, tag):
    mean_pred = np.repeat(Ytr.mean(axis=0, keepdims=True), Yte.shape[0], axis=0)
    mse_mean = float(np.mean((Yte - mean_pred) ** 2))
    r2_mean = _r2(Yte, mean_pred)

    clf = Ridge(alpha=1.0, random_state=42)
    clf.fit(Xtr, Ytr)
    Yhat = clf.predict(Xte)
    mse = float(np.mean((Yte - Yhat) ** 2))
    r2 = _r2(Yte, Yhat)

    T = Yte.shape[1] // dim
    per_frame_r2 = []
    for t in range(T):
        sl = slice(t * dim, (t + 1) * dim)
        per_frame_r2.append(_r2(Yte[:, sl], Yhat[:, sl]))

    # group R² over all frames (concat group dims across time)
    group_r2 = {}
    for gname, idxs in groups.items():
        cols = []
        for t in range(T):
            base = t * dim
            cols.extend([base + i for i in idxs])
        group_r2[gname] = _r2(Yte[:, cols], Yhat[:, cols])

    # future-only group R² (exclude t=0)
    group_r2_future = {}
    for gname, idxs in groups.items():
        cols = []
        for t in range(1, T):
            base = t * dim
            cols.extend([base + i for i in idxs])
        group_r2_future[gname] = _r2(Yte[:, cols], Yhat[:, cols]) if cols else None

    return {
        "feature": tag,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "feat_dim": int(Xtr.shape[1]),
        "target_dim": int(Ytr.shape[1]),
        "probe_mse": mse,
        "probe_r2": r2,
        "baseline_mean_mse": mse_mean,
        "baseline_mean_r2": r2_mean,
        "probe_beats_mean": bool(mse < mse_mean),
        "per_frame_r2": per_frame_r2,
        "frame0_r2": per_frame_r2[0] if per_frame_r2 else None,
        "future_r2_mean": float(np.mean(per_frame_r2[1:])) if len(per_frame_r2) > 1 else None,
        "group_r2_all_frames": group_r2,
        "group_r2_future_frames": group_r2_future,
    }


def random_proj(X, out_dim, seed):
    rng = np.random.RandomState(seed)
    proj = rng.randn(X.shape[1], out_dim).astype(np.float32) / np.sqrt(out_dim)
    return X @ proj


def conclude(probes: dict, decoder_frames: dict, decoder: str):
    """Fill teacher-facing conclusion fields."""
    bridge = probes.get("bridge_mean") or {}
    layer2 = probes.get("sf_mid_hidden_mean") or probes.get("mlp_hidden") or {}
    fut_r2 = bridge.get("future_r2_mean")
    beats = bridge.get("probe_beats_mean")
    present = bool(beats) and (fut_r2 is not None and fut_r2 > 0.0)

    g = bridge.get("group_r2_future_frames") or {}
    best_group = None
    if g:
        best_group = max(g.items(), key=lambda kv: (kv[1] if kv[1] is not None else -1e9))[0]

    layer_name = "sf_mid_hidden" if "sf_mid_hidden_mean" in probes else (
        "mlp_hidden" if "mlp_hidden" in probes else "bridge_only"
    )
    layer2_r2 = layer2.get("future_r2_mean")
    bridge_r2 = fut_r2
    if layer2_r2 is None:
        where = "bridge_tokens (no second-layer probe)"
    elif bridge_r2 is None:
        where = layer_name
    elif layer2_r2 > bridge_r2 + 0.02:
        where = f"{layer_name} > bridge"
    elif bridge_r2 > layer2_r2 + 0.02:
        where = f"bridge > {layer_name}"
    else:
        where = f"bridge ≈ {layer_name}"

    return {
        "future_info_in_representation": present,
        "evidence": {
            "bridge_future_r2": fut_r2,
            "bridge_probe_beats_mean_baseline": beats,
            "decoder_future_mse": decoder_frames.get("future_1_to_H"),
            "decoder_frame0_mse": decoder_frames.get("frame0"),
        },
        "which_layer": where,
        "which_state_group_best_future_r2": best_group,
        "group_future_r2": g,
        "notes": [
            "present=True only if bridge linear probe beats mean baseline AND future R²>0",
            "mean_pool has no true mid_hidden; mlp_hidden = Linear+GELU before final proj",
            f"decoder={decoder}",
        ],
    }


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path}", flush=True)
    model, model_config, collator, sf_dim, decoder = load_model(args)
    print(f"decoder={decoder} D_sf={sf_dim}", flush=True)
    singles, _ = build_singles(args, model_config, sf_dim)
    train_specs, test_specs, split_meta = sample_traj_steps(
        singles,
        args.num_train_trajs,
        args.num_test_trajs,
        args.steps_per_traj,
        args.seed,
    )
    print(
        f"traj split: train_trajs={split_meta['n_train_trajs']} "
        f"test_trajs={split_meta['n_test_trajs']} "
        f"train_steps={len(train_specs)} test_steps={len(test_specs)}",
        flush=True,
    )

    groups = group_slices(sf_dim)

    def run_split(specs):
        feats_mean, feats_flat = [], []
        mids_mean, mlps = [], []
        preds, gts, masks = [], [], []
        Ys = []
        meta = []
        for i0 in range(0, len(specs), args.batch_size):
            chunk = specs[i0 : i0 + args.batch_size]
            samples = []
            for s in chunk:
                ds = singles[s["task_i"]][1]
                samples.append(ds.get_step_data(s["traj_id"], s["base_index"]))
                # apply transforms
                samples[-1] = ds.transforms(samples[-1])
                meta.append(s)
            batch = collator(samples)
            feat, pred, gt, mask, mid, mlp_h = extract_batch(
                model, batch, args.device, decoder
            )
            feats_mean.append(feat.mean(dim=1).float().cpu().numpy())
            feats_flat.append(feat.float().cpu().numpy().reshape(feat.shape[0], -1))
            if mid is not None:
                mids_mean.append(mid.mean(dim=1).float().cpu().numpy())
            if mlp_h is not None:
                mlps.append(mlp_h.float().cpu().numpy())
            preds.append(pred.float().cpu())
            gts.append(gt.float().cpu())
            masks.append(mask.float().cpu() if mask is not None else None)
            gt_np = gt.float().cpu().numpy()
            if mask is not None:
                gt_np = gt_np * mask.float().cpu().numpy()
            Ys.append(gt_np.reshape(gt_np.shape[0], -1))
            print(f"  extracted {min(i0 + args.batch_size, len(specs))}/{len(specs)}", flush=True)
        pack = {
            "X_bridge_mean": np.concatenate(feats_mean, axis=0),
            "X_bridge_flat": np.concatenate(feats_flat, axis=0),
            "Y": np.concatenate(Ys, axis=0),
            "preds": preds,
            "gts": gts,
            "masks": masks,
            "meta": meta,
        }
        if mids_mean:
            pack["X_mid_mean"] = np.concatenate(mids_mean, axis=0)
        if mlps:
            pack["X_mlp"] = np.concatenate(mlps, axis=0)
        return pack

    print("=== train traj samples ===", flush=True)
    tr = run_split(train_specs)
    print("=== test traj samples ===", flush=True)
    te = run_split(test_specs)

    frame_stats = per_frame_mse(te["preds"], te["gts"], te["masks"])
    print("=== decoder per-frame MSE (test trajs) ===", flush=True)
    print(json.dumps(frame_stats, indent=2), flush=True)

    probes = {}
    probes["bridge_mean"] = fit_probe(
        tr["X_bridge_mean"], te["X_bridge_mean"], tr["Y"], te["Y"], sf_dim, groups, "bridge_mean"
    )
    Xtr_rp = random_proj(tr["X_bridge_flat"], 512, seed=0)
    Xte_rp = random_proj(te["X_bridge_flat"], 512, seed=0)
    probes["bridge_flat_rp512"] = fit_probe(
        Xtr_rp, Xte_rp, tr["Y"], te["Y"], sf_dim, groups, "bridge_flat_rp512"
    )
    if "X_mid_mean" in tr and "X_mid_mean" in te:
        probes["sf_mid_hidden_mean"] = fit_probe(
            tr["X_mid_mean"], te["X_mid_mean"], tr["Y"], te["Y"], sf_dim, groups, "sf_mid_hidden_mean"
        )
    if "X_mlp" in tr and "X_mlp" in te:
        probes["mlp_hidden"] = fit_probe(
            tr["X_mlp"], te["X_mlp"], tr["Y"], te["Y"], sf_dim, groups, "mlp_hidden"
        )

    print("=== linear probes ===", flush=True)
    print(json.dumps({k: {"r2": v["probe_r2"], "future_r2": v["future_r2_mean"], "groups_fut": v["group_r2_future_frames"]} for k, v in probes.items()}, indent=2), flush=True)

    pred_cat = torch.cat(te["preds"], dim=0).numpy().reshape(len(te["Y"]), -1)
    dec_mse = float(np.mean((pred_cat - te["Y"]) ** 2))
    conclusion = conclude(probes, frame_stats, decoder)

    report = {
        "model_path": args.model_path,
        "decoder": decoder,
        "state_future_dim": sf_dim,
        "split": {
            **split_meta,
            "scheme": "cross_trajectory",
            "steps_per_traj": args.steps_per_traj,
            "seed": args.seed,
            "n_train_steps": len(train_specs),
            "n_test_steps": len(test_specs),
        },
        "group_dim_indices_one_frame": groups,
        "decoder_per_frame_mse_test": frame_stats,
        "decoder_overall_mse_vs_gt_test": dec_mse,
        "linear_probe": probes,
        "conclusion": conclusion,
    }
    out_json = out / "sf_probe_normalized.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_json}", flush=True)
    print("=== conclusion ===", flush=True)
    print(json.dumps(conclusion, indent=2), flush=True)


if __name__ == "__main__":
    main()
