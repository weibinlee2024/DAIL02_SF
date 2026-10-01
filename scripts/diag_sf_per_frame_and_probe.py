#!/usr/bin/env python3
"""SF weight debug: per-frame SF loss + linear probe from frozen bridge tokens.

Teacher items:
1) Compare frame-0 vs future-frame SF MSE (t=0 can dominate the mean).
2) Freeze model; linear-probe future state from bridge tokens.
   Poor probe → info may not be in the representation.

Example:
  PYTHONPATH=. python scripts/diag_sf_per_frame_and_probe.py \\
    --model-path .../checkpoint-10000 \\
    --out-dir .../sf_weight_debug
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default="/media/weibin/weibin02/DIAL/20260922/pretrain-golden-sf-6task-10k-query-ca-tune-llm/checkpoint-10000",
    )
    p.add_argument(
        "--data-root",
        default="/media/weibin/Data/DIAL_DATASETS/LeRobot-AugPosRot-Correct",
    )
    p.add_argument(
        "--data-config",
        default="fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop",
    )
    p.add_argument("--out-dir", default="/media/weibin/weibin02/DIAL/20260922/sf_weight_debug")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_dataset(args, model_config):
    from gr00t.data.dataset import LeRobotSingleDatasetWithGoalImage, MultiEmbodimentLeRobotMixtureDataset
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
        singles.append(ds)

    if len(singles) == 1:
        return singles[0], data_config_cls

    mix = MultiEmbodimentLeRobotMixtureDataset(
        data_mixture=[(d, 1.0) for d in singles],
        mode="train",
        balance_dataset_weights=True,
        balance_trajectory_weights=True,
        seed=42,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
    return mix, data_config_cls


def load_model(args):
    from transformers import AutoConfig
    from gr00t.model.gr00t_n1_dial import GR00T_N1_5_DIAL
    from gr00t.model.transforms import DefaultDataCollator, BRIDGE_TOKENS

    model_config = AutoConfig.from_pretrained(args.model_path)
    num_bridge = model_config.bridge_cfg.get("num_bridge_tokens", 64)
    BRIDGE_TOKENS.update(num_bridge_tokens=num_bridge)
    tokenizer_len = model_config.bridge_cfg.get("tokenizer_len")
    if tokenizer_len is None:
        # Fallback: same default as bridge training configs
        tokenizer_len = 151729
    # Ensure SF is on when loading
    model = GR00T_N1_5_DIAL.from_pretrained(
        args.model_path,
        tokenizer_len=int(tokenizer_len),
        use_state_future=True,
        compute_state_future_loss=True,
        state_future_loss_weight=0.5,
        state_future_decoder=model_config.bridge_cfg.get("state_future_decoder", "query_ca"),
        state_future_dim=model_config.bridge_cfg.get("state_future_dim", 30),
        compute_bridge_loss=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)

    collator = DefaultDataCollator(vlm_path=model.config.backbone_cfg["vlm_path"])
    return model, model_config, collator


@torch.no_grad()
def extract_batch(model, batch, device):
    """Return bridge tokens [B,64,H], pred SF, gt SF, mask."""
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    backbone_inputs, action_inputs = model.prepare_input(batch)
    backbone_outputs = model.backbone(backbone_inputs)
    feat = backbone_outputs["backbone_features"][:, -model.num_bridge_tokens :]
    pred = model._predict_state_future(feat, feat.shape[0])
    gt = action_inputs.get("state_future")
    mask = action_inputs.get("state_future_mask")
    if gt is None or pred is None:
        raise RuntimeError("missing state_future GT or decoder")
    gt = gt.to(pred.dtype)
    if mask is not None:
        mask = mask.to(pred.dtype)
    return feat, pred, gt, mask


def per_frame_stats(preds, gts, masks):
    """preds/gts/masks: list of [B,T,D] tensors → per-frame MSE."""
    pred = torch.cat(preds, dim=0)
    gt = torch.cat(gts, dim=0)
    if masks[0] is None:
        mask = torch.ones_like(pred)
    else:
        mask = torch.cat(masks, dim=0)
    diff = (pred - gt) ** 2
    T = pred.shape[1]
    per_frame = []
    for t in range(T):
        m = mask[:, t]
        denom = m.sum().clamp_min(1.0)
        per_frame.append(float((diff[:, t] * m).sum() / denom))
    return {
        "per_frame_mse": per_frame,
        "frame0": per_frame[0],
        "future_1_to_H": float(np.mean(per_frame[1:])) if len(per_frame) > 1 else None,
        "near_1_4": float(np.mean(per_frame[1:5])) if len(per_frame) > 4 else None,
        "far_last4": float(np.mean(per_frame[-4:])) if len(per_frame) >= 4 else None,
        "all_frames": float(np.mean(per_frame)),
        "ratio_future_over_frame0": (
            float(np.mean(per_frame[1:]) / per_frame[0]) if per_frame[0] > 1e-12 else None
        ),
    }


def fit_linear_probe(X, Y, tag: str):
    """X: [N, F], Y: [N, T*D] flattened future state."""
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=42)
    # trivial baselines
    mean_pred = np.repeat(Ytr.mean(axis=0, keepdims=True), Yte.shape[0], axis=0)
    mse_mean = float(np.mean((Yte - mean_pred) ** 2))
    r2_mean = float(r2_score(Yte, mean_pred, multioutput="uniform_average"))

    clf = Ridge(alpha=1.0, random_state=42)
    clf.fit(Xtr, Ytr)
    Yhat = clf.predict(Xte)
    mse = float(np.mean((Yte - Yhat) ** 2))
    r2 = float(r2_score(Yte, Yhat, multioutput="uniform_average"))

    # per-frame R2 on probe
    T = Y.shape[1] // (Y.shape[1] // 17) if Y.shape[1] % 17 == 0 else None
    # We know D=30, T=17 typically
    D = 30
    T = Y.shape[1] // D
    per_frame_r2 = []
    for t in range(T):
        sl = slice(t * D, (t + 1) * D)
        per_frame_r2.append(float(r2_score(Yte[:, sl], Yhat[:, sl], multioutput="uniform_average")))

    return {
        "feature": tag,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "feat_dim": int(X.shape[1]),
        "target_dim": int(Y.shape[1]),
        "probe_mse": mse,
        "probe_r2": r2,
        "baseline_mean_mse": mse_mean,
        "baseline_mean_r2": r2_mean,
        "probe_beats_mean": mse < mse_mean,
        "per_frame_r2": per_frame_r2,
        "frame0_r2": per_frame_r2[0] if per_frame_r2 else None,
        "future_r2_mean": float(np.mean(per_frame_r2[1:])) if len(per_frame_r2) > 1 else None,
    }


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path}", flush=True)
    model, model_config, collator = load_model(args)
    dataset, _ = build_dataset(args, model_config)

    # sample indices
    n = min(args.num_samples, len(dataset))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset), size=n, replace=False)

    feats_mean = []
    feats_flat = []
    preds, gts, masks = [], [], []
    Y_list = []

    print(f"Extracting {n} samples, bs={args.batch_size} ...", flush=True)
    for i0 in range(0, n, args.batch_size):
        chunk = indices[i0 : i0 + args.batch_size]
        samples = [dataset[int(i)] for i in chunk]
        batch = collator(samples)
        feat, pred, gt, mask = extract_batch(model, batch, args.device)
        feats_mean.append(feat.mean(dim=1).float().cpu().numpy())
        feats_flat.append(feat.float().cpu().numpy().reshape(feat.shape[0], -1))
        preds.append(pred.float().cpu())
        gts.append(gt.float().cpu())
        masks.append(mask.float().cpu() if mask is not None else None)
        # probe target: flatten; zero-out invalid dims with mask mean-fill
        gt_np = gt.float().cpu().numpy()
        if mask is not None:
            m = mask.float().cpu().numpy()
            gt_np = gt_np * m  # invalid → 0
        Y_list.append(gt_np.reshape(gt_np.shape[0], -1))
        print(f"  {min(i0 + args.batch_size, n)}/{n}", flush=True)

    frame_stats = per_frame_stats(preds, gts, masks)
    print("=== per-frame SF MSE (decoder, frozen) ===", flush=True)
    print(json.dumps(frame_stats, indent=2), flush=True)

    X_mean = np.concatenate(feats_mean, axis=0)
    X_flat = np.concatenate(feats_flat, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    probe_mean = fit_linear_probe(X_mean, Y, tag="bridge_mean_pool")
    # flat 64*2048 is huge; subsample PCA-ish via random projection for tractability
    rng = np.random.RandomState(0)
    proj = rng.randn(X_flat.shape[1], 512).astype(np.float32) / np.sqrt(512)
    X_rp = X_flat @ proj
    probe_rp = fit_linear_probe(X_rp, Y, tag="bridge_flat_random_proj_512")

    print("=== linear probe ===", flush=True)
    print(json.dumps({"mean_pool": probe_mean, "random_proj": probe_rp}, indent=2), flush=True)

    # decoder vs probe on same targets (overall MSE)
    pred_cat = torch.cat(preds, dim=0).numpy().reshape(len(Y), -1)
    dec_mse = float(np.mean((pred_cat - Y) ** 2))

    report = {
        "model_path": args.model_path,
        "num_samples": n,
        "per_frame": frame_stats,
        "decoder_overall_mse_vs_gt": dec_mse,
        "linear_probe": {"mean_pool": probe_mean, "random_proj_512": probe_rp},
        "interpretation_hints": {
            "if_future_mse_much_gt_frame0": "t=0 dominates mean; report future frames separately",
            "if_probe_r2_near_baseline": "little linearly readable future-state info in bridge tokens",
            "if_probe_ok_but_action_hurt": "info present; lower w or change combine formula",
        },
    }
    out_json = out / "sf_per_frame_and_probe.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_json}", flush=True)

    # simple plot of per-frame mse
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(frame_stats["per_frame_mse"], marker="o", color="#7c3aed")
        ax.axhline(frame_stats["frame0"], color="#94a3b8", ls="--", label=f"frame0={frame_stats['frame0']:.4f}")
        ax.set_xlabel("frame index (0=current)")
        ax.set_ylabel("masked MSE")
        ax.set_title("SF decoder per-frame MSE (frozen ckpt)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png = out / "sf_per_frame_mse.png"
        fig.savefig(png, dpi=140)
        print(f"wrote {png}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
