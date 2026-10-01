"""Read LIBERO RND dense NPZ rollouts for DIAL LeRobot conversion.

Design contract (locked):
- RND exploration rollouts, NOT expert demonstrations.
- Keep success AND failure episodes (world-model / state coverage).
- Write per-episode `success` into LeRobot meta; do not drop failures by default.
- Optional success_only=True only for BC-only export ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

STATE_DIM = 8
ACTION_DIM = 7

SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
    "libero_10": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
}


@dataclass
class LiberoDenseNpzEpisode:
    """One aligned LIBERO dense episode ready for LeRobot writer."""

    state: np.ndarray  # (T, 8) float32
    action: np.ndarray  # (T, 7) float32
    agentview: np.ndarray  # (T, H, W, 3) uint8
    wrist: np.ndarray  # (T, H, W, 3) uint8
    language: str
    success: bool
    source_file: Path
    suite: str
    fps: float = 20.0

    @property
    def num_frames(self) -> int:
        return int(self.state.shape[0])


def _parse_success_from_filename(path: Path) -> Optional[bool]:
    name = path.name
    if "success=True" in name:
        return True
    if "success=False" in name:
        return False
    return None


def load_success_flag(npz_path: Path, data: np.lib.npyio.NpzFile) -> bool:
    if "success" in data.files:
        return bool(np.asarray(data["success"]).item())
    parsed = _parse_success_from_filename(npz_path)
    if parsed is not None:
        return parsed
    raise ValueError(f"Cannot determine success for {npz_path}")


def should_include_episode(success: bool, *, success_only: bool = False) -> bool:
    if success_only:
        return success
    return True


def iter_dense_npz_paths(
    root: Path,
    tag: str = "rnd_dense_all_v2",
    suites: Optional[list[str]] = None,
) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    if suites is None:
        suite_dirs = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    else:
        suite_dirs = tuple(SUITE_ALIASES.get(s, s) for s in suites)

    paths: list[Path] = []
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith("20"):
            continue
        for suite_name in suite_dirs:
            candidate = date_dir / suite_name / tag
            if candidate.is_dir():
                paths.extend(sorted(candidate.glob("*.npz")))
    if not paths:
        raise FileNotFoundError(f"No NPZ under {root} for tag={tag!r}, suites={suite_dirs}")
    return paths


def load_dense_npz_episode(npz_path: Path, fps: float = 20.0) -> LiberoDenseNpzEpisode:
    with np.load(npz_path, allow_pickle=True) as data:
        success = load_success_flag(npz_path, data)
        proprio = np.asarray(data["proprio"], dtype=np.float32)
        action = np.asarray(data["actions_env"], dtype=np.float32)
        agentview = np.asarray(data["rgb"], dtype=np.uint8)
        wrist = np.asarray(data["wrist_rgb"], dtype=np.uint8)
        language = str(np.asarray(data["task_description"]).item()).strip()
        if "suite" in data.files:
            suite = str(np.asarray(data["suite"]).item())
        else:
            suite = npz_path.parent.parent.name

    if proprio.ndim != 2 or proprio.shape[1] != STATE_DIM:
        raise ValueError(f"{npz_path}: unexpected proprio shape {proprio.shape}")
    if action.ndim != 2 or action.shape[1] < ACTION_DIM:
        raise ValueError(f"{npz_path}: unexpected action shape {action.shape}")
    action = action[:, :ACTION_DIM]

    t = min(proprio.shape[0] - 1, action.shape[0], agentview.shape[0] - 1, wrist.shape[0] - 1)
    if t < 2:
        raise ValueError(f"{npz_path}: too few aligned frames ({t})")

    return LiberoDenseNpzEpisode(
        state=proprio[:t].astype(np.float32),
        action=action[:t].astype(np.float32),
        agentview=agentview[:t].astype(np.uint8),
        wrist=wrist[:t].astype(np.uint8),
        language=language,
        success=success,
        source_file=Path(npz_path),
        suite=suite,
        fps=float(fps),
    )


def iter_dense_npz_episodes(
    root: Path,
    tag: str = "rnd_dense_all_v2",
    suites: Optional[list[str]] = None,
    success_only: bool = False,
    max_episodes: Optional[int] = None,
) -> Iterator[LiberoDenseNpzEpisode]:
    """
    Yield episodes from RND dense NPZ.

    success_only defaults to False: keep failure trajectories for world-model coverage.
    """
    count = 0
    skipped_fail = 0
    for npz_path in iter_dense_npz_paths(root, tag=tag, suites=suites):
        ep = load_dense_npz_episode(npz_path)
        if not should_include_episode(ep.success, success_only=success_only):
            skipped_fail += 1
            continue
        yield ep
        count += 1
        if max_episodes is not None and count >= max_episodes:
            break
    if count == 0:
        raise RuntimeError(
            f"No episodes yielded from {root} (success_only={success_only}, "
            f"skipped_failures={skipped_fail}). Default keeps all episodes."
        )
