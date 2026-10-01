#!/usr/bin/env python3
"""
Convert LIBERO RND dense NPZ to GR00T/DIAL-compatible LeRobot v2.0.

Default: keep ALL episodes (success + failure) for world-model coverage.
Writes per-episode `success` into meta/episodes.jsonl.

Example (smoke):
  python preprocessing/convert_libero_dense_npz_to_lerobot.py \\
      --npz_root "/path/to/VLA-RFT/rollouts/rl_dense_tokenize" \\
      --output_dir data/libero/libero_spatial \\
      --suites spatial --max_episodes 20

Full suite:
  python preprocessing/convert_libero_dense_npz_to_lerobot.py \\
      --npz_root "/path/to/rl_dense_tokenize" \\
      --output_dir data/libero/libero_spatial \\
      --suites spatial --tag rnd_dense_all_v2
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tyro
from tqdm import tqdm

_PREPROCESS_DIR = Path(__file__).resolve().parent
if str(_PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS_DIR))

from libero_dense_npz_io import (  # noqa: E402
    SUITE_ALIASES,
    iter_dense_npz_episodes,
    iter_dense_npz_paths,
)
from lerobot_dataset_libero import LiberoLeRobotWriter, reset_output_dir  # noqa: E402


@dataclass
class Args:
    npz_root: str
    """Root of rl_dense_tokenize (contains date/libero_*/tag/*.npz)."""

    output_dir: str
    """Output LeRobot v2.0 directory (one suite recommended per call)."""

    tag: str = "rnd_dense_all_v2"
    """RND collection tag."""

    suites: Optional[List[str]] = None
    """Suite aliases: spatial, object, goal, 10. Default: all four."""

    fps: float = 20.0
    image_size: int = 128
    """Declared video size in info.json (NPZ is 128)."""

    max_episodes: Optional[int] = None
    """Cap for smoke tests."""

    success_only: bool = False
    """If True, drop failures (BC-only ablation). Default False = keep all."""

    resume: bool = False
    overwrite: bool = False
    robot_type: str = "libero_franka"


def main(args: Args) -> None:
    npz_root = Path(args.npz_root)
    output_dir = Path(args.output_dir)

    if not npz_root.is_dir():
        raise FileNotFoundError(f"npz_root not found: {npz_root}")

    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")

    if args.overwrite and output_dir.exists():
        print(f"[convert] removing existing output: {output_dir}")
        reset_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    paths = iter_dense_npz_paths(npz_root, tag=args.tag, suites=args.suites)
    total_hint = len(paths)
    if args.max_episodes is not None:
        total_hint = min(total_hint, args.max_episodes)
    print(
        f"[convert] found {len(paths)} npz under {npz_root} "
        f"(tag={args.tag}, suites={args.suites}, success_only={args.success_only})"
    )

    writer = LiberoLeRobotWriter(
        root=output_dir,
        fps=args.fps,
        image_size=args.image_size,
        robot_type=args.robot_type,
        resume=args.resume,
    )

    written = 0
    n_succ = 0
    n_fail = 0
    for ep in tqdm(
        iter_dense_npz_episodes(
            npz_root,
            tag=args.tag,
            suites=args.suites,
            success_only=args.success_only,
            max_episodes=args.max_episodes,
        ),
        total=total_hint,
        desc="NPZ->LeRobot",
    ):
        writer.save_episode(ep, success=ep.success)
        written += 1
        if ep.success:
            n_succ += 1
        else:
            n_fail += 1

    if written == 0:
        raise RuntimeError(f"No episodes written from {npz_root}")

    writer.finalize()
    print(
        f"[convert] done: {written} episodes "
        f"(success={n_succ}, fail={n_fail}) -> {output_dir}"
    )
    print(
        f"[convert] next: python preprocessing/validate_libero_lerobot.py "
        f"--dataset_path {output_dir}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
