#!/usr/bin/env python3
"""Verify the LIBERO-plus task-level train/held-out split.

Checks:
  1. train/held-out task_index sets are disjoint and cover the suite's tasks;
  2. episode counts per task;
  3. LeRobotSingleDataset(task_indices=train) contains only train episodes;
  4. LeRobotSingleDataset(task_indices=heldout) contains only held-out episodes;
  5. train/held-out episode-id sets do not overlap.

Example:
  python scripts/verify_libero_plus_task_split.py \
    --suite-path /media/.../Sylvest/libero_plus_spatial
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr00t.data.dataset import LeRobotSingleDatasetWithGoalImage  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.experiment.data_config_dial import load_data_config  # noqa: E402


def parse_indices(raw: str) -> list[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x.strip()]


def load_split_metadata(suite_path: Path):
    with (suite_path / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        episodes = [json.loads(line) for line in handle if line.strip()]
    with (suite_path / "meta" / "tasks.jsonl").open("r", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]

    task_text_to_index = {row["task"]: int(row["task_index"]) for row in tasks}
    episodes_by_task: dict[int, list[int]] = {row["task_index"]: [] for row in tasks}
    episode_task_indices: dict[int, list[int]] = {}
    unknown_tasks = Counter()

    for episode in episodes:
        indices = []
        for task_text in episode.get("tasks", []):
            if task_text not in task_text_to_index:
                unknown_tasks[task_text] += 1
                continue
            indices.append(task_text_to_index[task_text])
        episode_task_indices[int(episode["episode_index"])] = indices
        for task_index in indices:
            episodes_by_task[task_index].append(int(episode["episode_index"]))

    if unknown_tasks:
        raise KeyError(f"tasks not present in tasks.jsonl: {dict(unknown_tasks)}")
    return episodes, tasks, episode_task_indices, episodes_by_task


def expected_episode_ids(
    episode_task_indices: dict[int, list[int]], allowed: set[int]
) -> set[int]:
    return {
        episode_id
        for episode_id, indices in episode_task_indices.items()
        if indices and all(index in allowed for index in indices)
    }


def check_dataset(
    suite_path: Path,
    task_indices: list[int],
    expected_ids: set[int],
    video_backend: str,
) -> set[int]:
    cfg = load_data_config("libero")
    dataset = LeRobotSingleDatasetWithGoalImage(
        dataset_path=suite_path,
        modality_configs=cfg.modality_config(),
        embodiment_tag=EmbodimentTag("libero"),
        transforms=None,
        video_backend=video_backend,
        split="[:]",
        temporal_split=None,
        task_indices=task_indices,
    )
    actual_ids = {int(x) for x in dataset.trajectory_ids.tolist()}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise AssertionError(
            f"dataset mismatch for task_indices={task_indices}: "
            f"missing={missing} extra={extra}"
        )
    print(
        f"[dataset] task_indices={task_indices}: "
        f"{len(actual_ids)} episodes, {len(dataset)} steps"
    )
    return actual_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-path", required=True)
    parser.add_argument(
        "--train-task-indices",
        default="0 1 2 3 4 5 6 7",
        help="Whitespace/comma separated task_index values.",
    )
    parser.add_argument(
        "--heldout-task-indices",
        default="8 9",
        help="Whitespace/comma separated task_index values.",
    )
    parser.add_argument(
        "--video-backend",
        default="torchvision_av",
        choices=["decord", "torchvision_av"],
    )
    args = parser.parse_args()

    suite_path = Path(args.suite_path).expanduser().resolve()
    if not suite_path.is_dir():
        raise FileNotFoundError(suite_path)

    train = parse_indices(args.train_task_indices)
    heldout = parse_indices(args.heldout_task_indices)
    train_set = set(train)
    heldout_set = set(heldout)

    if len(train_set) != len(train) or len(heldout_set) != len(heldout):
        raise ValueError("task_indices contain duplicates")
    overlap = train_set & heldout_set
    if overlap:
        raise ValueError(f"train/held-out task_indices overlap: {sorted(overlap)}")

    episodes, tasks, episode_task_indices, episodes_by_task = load_split_metadata(suite_path)
    all_task_indices = {int(row["task_index"]) for row in tasks}
    if train_set | heldout_set != all_task_indices:
        raise ValueError(
            "train+held-out does not cover all tasks: "
            f"missing={sorted(all_task_indices - (train_set | heldout_set))}, "
            f"extra={sorted((train_set | heldout_set) - all_task_indices)}"
        )

    print(f"[suite] {suite_path.name}")
    print(f"[split] train={sorted(train_set)} (80%), held-out={sorted(heldout_set)} (20%)")
    print(f"[episodes] total={len(episodes)}")
    rows = []
    for task_index in sorted(all_task_indices):
        n_eps = len(episodes_by_task[task_index])
        n_frames = sum(
            episode["length"]
            for episode in episodes
            if task_index in episode_task_indices[int(episode["episode_index"])]
        )
        split = "train" if task_index in train_set else "held-out"
        rows.append((task_index, n_eps, n_frames, split))
        print(f"  task {task_index}: episodes={n_eps:4d} frames={n_frames:7d}  {split}")

    train_eps = expected_episode_ids(episode_task_indices, train_set)
    heldout_eps = expected_episode_ids(episode_task_indices, heldout_set)
    if train_eps & heldout_eps:
        raise AssertionError(
            f"episode leakage: {sorted(train_eps & heldout_eps)[:20]}"
        )
    print(
        f"[episode ids] train={len(train_eps)} held-out={len(heldout_eps)} "
        f"overlap={len(train_eps & heldout_eps)}"
    )

    train_actual = check_dataset(
        suite_path, train, train_eps, args.video_backend
    )
    heldout_actual = check_dataset(
        suite_path, heldout, heldout_eps, args.video_backend
    )
    if train_actual & heldout_actual:
        raise AssertionError("dataset episode ids overlap between train and held-out")
    print("[PASS] task-level split is clean; no episode overlap")


if __name__ == "__main__":
    main()
