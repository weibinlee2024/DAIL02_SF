#!/usr/bin/env python3
"""Train WM-as-Verifier dynamics head independently (does not touch System-1/2).

(s_t, a_chunk_normalized) → ŝ[17, 18]  pos+rot6d only.
Same offline LeRobot data + data_config state_future / action stats.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import tyro
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gr00t.model.wm_verifier.action_conditioned_dynamics import (
    POS_ROT_DIM,
    ActionConditionedDynamics,
    GroupNormMSE,
)
from gr00t.model.wm_verifier.data import (
    build_wm_train_dataset,
    infer_action_dim,
    wm_collate,
)

DEFAULT_TASKS = [
    "PnPBottleToCabinetClose",
    "PnPCanToDrawerClose",
    "PnPCupToDrawerClose",
    "PnPMilkToMicrowaveClose",
    "PnPPotatoToMicrowaveClose",
    "PnPWineToCabinetClose",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA",
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
    "PosttrainPnPNovelFromPlacematToBasketSplitA",
    "PosttrainPnPNovelFromPlacematToBowlSplitA",
    "PosttrainPnPNovelFromPlacematToPlateSplitA",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
    "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA",
    "PosttrainPnPNovelFromPlateToPanSplitA",
    "PosttrainPnPNovelFromPlateToPlateSplitA",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "PosttrainPnPNovelFromTrayToPlateSplitA",
    "PosttrainPnPNovelFromTrayToPotSplitA",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
]


@dataclass
class Args:
    dataset_path: Optional[List[str]] = None
    data_root: str = "/home/weibin02/DIAL_DATASETS/LeRobot-AugPosRot-Correct"
    data_config: str = "fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop"
    data_split: str = "[:-10]"
    output_dir: str = "outputs/wm_verifier_posrot18"
    max_steps: int = 20000
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    num_layers: int = 3
    num_workers: int = 4
    log_every: int = 50
    save_every: int = 5000
    calib_batches: int = 64
    device: str = "cuda"
    seed: int = 42


def _resolve_paths(args: Args) -> list[str]:
    if args.dataset_path:
        return list(args.dataset_path)
    paths = [os.path.join(args.data_root, f"gr1_unified.{t}") for t in DEFAULT_TASKS]
    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(f"missing datasets (first): {missing[0]}")
    return paths


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = _resolve_paths(args)
    print(f"[WM] datasets={len(paths)} out={out}", flush=True)
    dataset, meta = build_wm_train_dataset(
        dataset_paths=paths,
        data_config=args.data_config,
        data_split=args.data_split,
    )
    action_dim = infer_action_dim(dataset)
    meta["action_dim"] = action_dim
    print(f"[WM] meta={json.dumps(meta)}", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=wm_collate,
        drop_last=True,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ActionConditionedDynamics(
        state_dim=POS_ROT_DIM,
        action_dim=action_dim,
        action_horizon=int(meta["action_horizon"]),
        state_horizon=int(meta["state_horizon"]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    crit = GroupNormMSE(state_dim=POS_ROT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Calibrate group std from GT (normalized state_future space).
    print(f"[WM] calibrating group std on {args.calib_batches} batches ...", flush=True)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            crit.update_stats(batch["s_future"].to(device), batch["s_mask"].to(device))
            if i + 1 >= args.calib_batches:
                break
    print(
        f"[WM] pos_std={crit.pos_moment.sqrt().item():.4f} "
        f"rot_std={crit.rot_moment.sqrt().item():.4f}",
        flush=True,
    )

    (out / "wm_meta.json").write_text(
        json.dumps(
            {
                **meta,
                "pos_std": float(crit.pos_moment.sqrt().item()),
                "rot_std": float(crit.rot_moment.sqrt().item()),
                "max_steps": args.max_steps,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
            },
            indent=2,
        )
    )

    model.train()
    step = 0
    it = iter(loader)
    while step < args.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        s_t = batch["s_t"].to(device, non_blocking=True)
        a = batch["a_chunk"].to(device, non_blocking=True)
        gt = batch["s_future"].to(device, non_blocking=True)
        mask = batch["s_mask"].to(device, non_blocking=True)

        pred = model(s_t, a)
        out_dict = crit(pred, gt, mask)
        loss = out_dict["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1

        if step % args.log_every == 0 or step == 1:
            print(
                f"[WM] step={step}/{args.max_steps} "
                f"loss={loss.item():.4f} "
                f"score={out_dict['loss_score'].item():.4f} "
                f"aux={out_dict['loss_aux'].item():.4f} "
                f"pos_s={out_dict['loss_pos_score'].item():.4f} "
                f"rot_s={out_dict['loss_rot_score'].item():.4f}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.max_steps:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "crit": crit.state_dict(),
                "meta": meta,
                "args": vars(args),
            }
            path = out / f"checkpoint-{step}.pt"
            torch.save(ckpt, path)
            torch.save(ckpt, out / "checkpoint-last.pt")
            print(f"[WM] saved {path}", flush=True)

    print("[WM] done", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
