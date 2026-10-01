#!/usr/bin/env python3
"""Validate a converted LIBERO LeRobot v2.0 dataset for DIAL loading."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

_PREPROCESS_DIR = Path(__file__).resolve().parent
_ROOT = _PREPROCESS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class Args:
    dataset_path: str
    """LeRobot v2.0 root produced by convert_libero_dense_npz_to_lerobot.py."""

    try_dial_loader: bool = False
    """If True, instantiate LeRobotSingleDataset (needs gr00t + deps)."""


def main(args: Args) -> None:
    root = Path(args.dataset_path)
    meta = root / "meta"
    required = [
        meta / "info.json",
        meta / "modality.json",
        meta / "stats.json",
        meta / "episodes.jsonl",
        meta / "tasks.jsonl",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    with open(meta / "info.json") as f:
        info = json.load(f)
    with open(meta / "modality.json") as f:
        modality = json.load(f)

    episodes = []
    with open(meta / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))

    n_succ = sum(1 for e in episodes if e.get("success") is True)
    n_fail = sum(1 for e in episodes if e.get("success") is False)
    print(f"[ok] episodes={len(episodes)} success={n_succ} fail={n_fail}")
    print(f"[ok] info.total_episodes={info.get('total_episodes')} fps={info.get('fps')}")
    print(f"[ok] modality.state keys={list(modality['state'].keys())}")
    print(f"[ok] modality.action keys={list(modality['action'].keys())}")

    if not episodes:
        raise RuntimeError("empty episodes.jsonl")

    ep0 = episodes[0]["episode_index"]
    chunk = ep0 // info["chunks_size"]
    pq = root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep0:06d}.parquet"
    v0 = (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.image"
        / f"episode_{ep0:06d}.mp4"
    )
    v1 = (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.wrist_image"
        / f"episode_{ep0:06d}.mp4"
    )
    for p in (pq, v0, v1):
        if not p.exists():
            raise FileNotFoundError(p)
    print(f"[ok] sample episode files exist: {pq.name}")

    import pandas as pd

    df = pd.read_parquet(pq)
    state = np.stack(df["observation.state"].to_numpy())
    action = np.stack(df["action"].to_numpy())
    assert state.shape[1] == 8, state.shape
    assert action.shape[1] == 7, action.shape
    assert len(df) == episodes[0]["length"], (len(df), episodes[0]["length"])
    print(f"[ok] parquet shapes state={state.shape} action={action.shape}")

    if args.try_dial_loader:
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.experiment.data_config_dial import load_data_config

        cfg = load_data_config("libero", use_bridge=True)
        ds = LeRobotSingleDataset(
            dataset_path=root,
            modality_configs=cfg.modality_config(),
            embodiment_tag="libero",
            transforms=None,
        )
        sample = ds[0]
        print(f"[ok] DIAL loader sample keys={list(sample.keys())[:8]}... len={len(ds)}")

    print("[validate] PASS")


if __name__ == "__main__":
    main(tyro.cli(Args))
