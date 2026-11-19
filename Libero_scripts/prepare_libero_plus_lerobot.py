#!/usr/bin/env python3
"""Prepare official LIBERO-plus LeRobot v2.1 suites for DIAL/GR00T.

The LIBERO-plus LeRobot release uses:
  - observation.images.front / observation.images.wrist
  - observation.state (8D) / action (7D)
  - meta/episodes_stats.jsonl (per-episode stats), no meta/modality.json,
    no dataset-level meta/stats.json

DIAL's LeRobot reader expects meta/modality.json and meta/stats.json. This
script adds both in-place (idempotent unless --overwrite):
  - maps video.image -> observation.images.front
  - maps video.wrist_image -> observation.images.wrist
  - aggregates episodes_stats.jsonl into dataset-level stats.json
  - q01/q99 are set to min/max because LiberoDataConfig uses min_max
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODALITY = {
    "state": {
        "eef_position": {
            "original_key": "observation.state",
            "start": 0,
            "end": 3,
        },
        "eef_rotation": {
            "original_key": "observation.state",
            "start": 3,
            "end": 6,
            "rotation_type": "axis_angle",
        },
        "gripper_position": {
            "original_key": "observation.state",
            "start": 6,
            "end": 8,
        },
    },
    "action": {
        "eef_position_delta": {
            "original_key": "action",
            "start": 0,
            "end": 3,
            "absolute": False,
        },
        "eef_rotation_delta": {
            "original_key": "action",
            "start": 3,
            "end": 6,
            "rotation_type": "axis_angle",
            "absolute": False,
        },
        "gripper_position": {
            "original_key": "action",
            "start": 6,
            "end": 7,
        },
    },
    "video": {
        "image": {"original_key": "observation.images.front"},
        "wrist_image": {"original_key": "observation.images.wrist"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}

STAT_KEYS = ("observation.state", "action")


def _as_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _load_episode_task_filter(suite_path: Path, task_indices: set[int] | None):
    """Return episode_id -> keep flag for the requested task_index whitelist."""
    if task_indices is None:
        return None

    tasks_path = suite_path / "meta" / "tasks.jsonl"
    episodes_path = suite_path / "meta" / "episodes.jsonl"
    with tasks_path.open("r", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    task_text_to_index = {row["task"]: int(row["task_index"]) for row in tasks}
    with episodes_path.open("r", encoding="utf-8") as handle:
        episodes = [json.loads(line) for line in handle if line.strip()]

    keep = {}
    for episode in episodes:
        indices = [
            task_text_to_index[task]
            for task in episode.get("tasks", [])
            if task in task_text_to_index
        ]
        keep[int(episode["episode_index"])] = bool(indices) and all(
            index in task_indices for index in indices
        )
    return keep


def aggregate_stats(
    suite_path: Path,
    task_indices: set[int] | None = None,
) -> dict:
    episodes_stats_path = suite_path / "meta" / "episodes_stats.jsonl"
    if not episodes_stats_path.exists():
        raise FileNotFoundError(f"missing {episodes_stats_path}")
    episode_filter = _load_episode_task_filter(suite_path, task_indices)
    included_episodes = 0
    total_episodes = 0

    acc = {
        key: {
            "min": None,
            "max": None,
            "sum": None,
            "sum_sq": None,
            "count": 0,
        }
        for key in STAT_KEYS
    }

    with episodes_stats_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            total_episodes += 1
            if episode_filter is not None:
                episode_id = int(episode.get("episode_index", -1))
                if not episode_filter.get(episode_id, False):
                    continue
            included_episodes += 1
            stats = episode.get("stats", {})
            for key in STAT_KEYS:
                if key not in stats:
                    continue
                entry = stats[key]
                count = int(_as_vector(entry["count"])[0])
                if count <= 0:
                    continue
                minimum = _as_vector(entry["min"])
                maximum = _as_vector(entry["max"])
                mean = _as_vector(entry["mean"])
                std = _as_vector(entry["std"])

                if acc[key]["min"] is None:
                    acc[key]["min"] = minimum.copy()
                    acc[key]["max"] = maximum.copy()
                    acc[key]["sum"] = mean * count
                    acc[key]["sum_sq"] = (std**2 + mean**2) * count
                else:
                    acc[key]["min"] = np.minimum(acc[key]["min"], minimum)
                    acc[key]["max"] = np.maximum(acc[key]["max"], maximum)
                    acc[key]["sum"] += mean * count
                    acc[key]["sum_sq"] += (std**2 + mean**2) * count
                acc[key]["count"] += count

    print(
        f"[stats] {suite_path.name}: included {included_episodes}/{total_episodes} episodes; "
        f"task_indices={sorted(task_indices) if task_indices is not None else 'all'}"
    )
    out = {}
    for key in STAT_KEYS:
        item = acc[key]
        if item["count"] == 0 or item["sum"] is None:
            raise RuntimeError(f"no stats found for {key} in {episodes_stats_path}")
        mean = item["sum"] / item["count"]
        var = item["sum_sq"] / item["count"] - mean**2
        std = np.sqrt(np.maximum(var, 0.0))
        out[key] = {
            "min": item["min"].tolist(),
            "max": item["max"].tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            # q01/q99 are not used by min_max normalization; keep the schema valid.
            "q01": item["min"].tolist(),
            "q99": item["max"].tolist(),
        }
    return out


def prepare_suite(
    suite_path: Path,
    overwrite: bool = False,
    task_indices: set[int] | None = None,
) -> None:
    meta_dir = suite_path / "meta"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"missing meta dir: {meta_dir}")

    modality_path = meta_dir / "modality.json"
    stats_path = meta_dir / "stats.json"

    if overwrite or not modality_path.exists():
        with modality_path.open("w", encoding="utf-8") as handle:
            json.dump(MODALITY, handle, indent=4)
        print(f"[prepare] wrote {modality_path}")
    else:
        print(f"[prepare] keep existing {modality_path}")

    if overwrite or not stats_path.exists():
        stats = aggregate_stats(suite_path, task_indices=task_indices)
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=4)
        print(f"[prepare] wrote {stats_path} ({', '.join(STAT_KEYS)})")
    else:
        print(f"[prepare] keep existing {stats_path}")

    info_path = suite_path / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        print(
            f"[prepare] {suite_path.name}: "
            f"episodes={info.get('total_episodes')} frames={info.get('total_frames')} "
            f"tasks={info.get('total_tasks')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-path",
        action="append",
        required=True,
        help="LIBERO-plus suite root containing meta/ and data/ (repeatable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing modality.json / stats.json.",
    )
    parser.add_argument(
        "--task-indices",
        default="",
        help=(
            "Optional task_index whitelist for stats.json, whitespace/comma separated. "
            "Use the training split (e.g. '0 1 2 3 4 5 6 7') to avoid held-out "
            "tasks influencing normalization statistics."
        ),
    )
    args = parser.parse_args()

    task_indices = None
    if args.task_indices.strip():
        task_indices = {
            int(x) for x in args.task_indices.replace(",", " ").split() if x.strip()
        }

    for raw_path in args.suite_path:
        suite_path = Path(raw_path).expanduser().resolve()
        prepare_suite(
            suite_path,
            overwrite=args.overwrite,
            task_indices=task_indices,
        )


if __name__ == "__main__":
    main()
