#!/usr/bin/env python3
"""Build LIBERO-plus held-out manifests for task-level 8/2 evaluation.

Held-out base tasks per suite: task_index 8 and 9 (20% of the 10 tasks).
Benchmark variants are selected by matching the base task slug against
``task_classification.json`` names. Produces:
  - heldout_full.json: all held-out benchmark variants (1,996)
  - pilot_v1.json: 1 variant per category per held-out base task (56)
  - pilot_v5.json: 5 variants per category per held-out base task (280)

Run in the ``libero-plus`` env so the benchmark task order is available.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

SUITE_MAP = {
    "libero_spatial": "libero_plus_spatial",
    "libero_object": "libero_plus_object",
    "libero_goal": "libero_plus_goal",
    "libero_10": "libero_plus_10",
}
MAX_STEPS = {
    "libero_spatial": 600,
    "libero_object": 600,
    "libero_goal": 600,
    "libero_10": 700,
}
CATEGORIES = [
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
]


def _slug(text: str) -> str:
    return "_".join(text.lower().replace(",", "").replace(".", "").split())


def _select_evenly(items: list, k: int) -> list:
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[min(int(i * step), len(items) - 1)] for i in range(k)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root",
        default="/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/Libero_plus/lerobot/root/.cache/huggingface/lerobot/Sylvest",
    )
    parser.add_argument(
        "--classification",
        default="/home/weibin02/LIBERO-plus/libero/libero/benchmark/task_classification.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/weibin02/dca4b98f-3ef4-4bfa-91a4-90338de65e8a/home/weibin02/DIAL02/outputs/20261112/libero_plus_eval_manifests",
    )
    parser.add_argument("--heldout-task-indices", default="8 9")
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()

    os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path.home() / ".libero-openwam-plus"))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    from libero.libero import benchmark

    suite_root = Path(args.suite_root).expanduser().resolve()
    classification = json.loads(Path(args.classification).read_text(encoding="utf-8"))
    heldout_task_indices = [int(x) for x in args.heldout_task_indices.split()]

    full: list[dict] = []
    pilot_k: dict[int, list[dict]] = {1: [], 5: []}
    summary = {}

    for bench_suite, data_suite in SUITE_MAP.items():
        suite_path = suite_root / data_suite
        tasks = {
            row["task_index"]: row["task"]
            for row in (
                json.loads(line)
                for line in (suite_path / "meta" / "tasks.jsonl").open(encoding="utf-8")
                if line.strip()
            )
        }
        suite_obj = benchmark.get_benchmark_dict()[bench_suite]()
        names = suite_obj.get_task_names()
        name_to_index = {name: idx for idx, name in enumerate(names)}
        bench_entries = classification[bench_suite]

        suite_all = []
        suite_summary = Counter()
        for base_index in heldout_task_indices:
            base_task = tasks[base_index]
            slug = _slug(base_task)
            matches = [row for row in bench_entries if slug in row["name"]]
            if not matches:
                raise RuntimeError(
                    f"no benchmark matches for {bench_suite} task {base_index}: {base_task!r}"
                )
            matches = sorted(matches, key=lambda row: row["name"])
            for row in matches:
                if row["name"] not in name_to_index:
                    raise KeyError(f"{row['name']!r} not in {bench_suite} task names")
                item = {
                    "suite": bench_suite,
                    "suite_data_dir": data_suite,
                    "base_task_index": base_index,
                    "base_task": base_task,
                    "category": row.get("category"),
                    "difficulty_level": row.get("difficulty_level"),
                    "name": row["name"],
                    "benchmark_index": name_to_index[row["name"]],
                    "max_steps": MAX_STEPS[bench_suite],
                }
                suite_all.append(item)
                full.append(item)
                suite_summary[row.get("category")] += 1

        # Stratified pilots: choose evenly spaced variants per (base_task, category).
        for base_index in heldout_task_indices:
            base_items = [item for item in suite_all if item["base_task_index"] == base_index]
            for category in CATEGORIES:
                cat_items = sorted(
                    [item for item in base_items if item["category"] == category],
                    key=lambda item: item["name"],
                )
                for k in pilot_k:
                    chosen = _select_evenly(cat_items, k)
                    pilot_k[k].extend(chosen)

        summary[bench_suite] = {
            "all_heldout": len(suite_all),
            "by_category": dict(sorted(suite_summary.items())),
            "heldout_task_indices": heldout_task_indices,
        }

    # Deduplicate (a name should only match one base task, but keep it safe).
    def _dedupe(items: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for item in items:
            key = (item["suite"], item["name"])
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    full = _dedupe(full)
    pilot_k = {k: _dedupe(v) for k, v in pilot_k.items()}

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "heldout_full.json": full,
        "pilot_v1.json": pilot_k[1],
        "pilot_v5.json": pilot_k[5],
    }
    for filename, items in outputs.items():
        path = out_dir / filename
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        print(f"[manifest] {filename}: {len(items)} tasks -> {path}")

    summary["total"] = {
        "heldout_full": len(full),
        "pilot_v1": len(pilot_k[1]),
        "pilot_v5": len(pilot_k[5]),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
