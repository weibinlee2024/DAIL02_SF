#!/usr/bin/env python3
"""Run a LIBERO-plus manifest against the DIAL ZMQ policy server.

Manifest entries are produced by ``build_libero_plus_heldout_manifest.py``.
One server stays alive for the whole run; each task gets one episode with the
official settings (1 trial, settle 30, 600/700 steps, seed 10000).

Run in the ``libero-plus`` conda env.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_libero_plus_smoke import (  # noqa: E402
    ZMQPolicyClient,
    _resolve_bddl,
    _rotate_180_if_requested,
)
from gr00t.eval.libero_obs_action import (  # noqa: E402
    dial_action_chunk_to_libero,
    libero_obs_to_dial,
)


def _load_done_keys(results_path: Path) -> set[tuple[str, str]]:
    done = set()
    if not results_path.exists():
        return done
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row.get("suite", ""), row.get("name", "")))
    return done


def _summary(rows: list[dict]) -> dict:
    def _agg(key_fn):
        buckets = defaultdict(lambda: [0, 0])
        for row in rows:
            key = key_fn(row)
            buckets[key][1] += 1
            buckets[key][0] += int(bool(row.get("success")))
        return {
            str(key): {
                "success": value[0],
                "total": value[1],
                "rate": value[0] / value[1] if value[1] else 0.0,
            }
            for key, value in sorted(buckets.items(), key=lambda item: str(item[0]))
        }

    return {
        "success": int(sum(bool(row.get("success")) for row in rows)),
        "total": len(rows),
        "rate": (
            sum(bool(row.get("success")) for row in rows) / len(rows) if rows else 0.0
        ),
        "by_suite": _agg(lambda row: row.get("suite")),
        "by_category": _agg(lambda row: row.get("category")),
        "by_base_task": _agg(
            lambda row: f"{row.get('suite')}:task{row.get('base_task_index')}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5582)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 = all remaining")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rotate-180", action="store_true", default=True)
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false")
    parser.add_argument("--save-actions", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path.home() / ".libero-openwam-plus"))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    random.seed(args.seed)
    np.random.seed(args.seed)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_dir = Path(args.result_dir).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    results_path = result_dir / "results.jsonl"
    done_keys = _load_done_keys(results_path) if args.resume else set()

    items = manifest[args.start_index :]
    if args.max_tasks and args.max_tasks > 0:
        items = items[: args.max_tasks]
    if args.resume:
        items = [
            item
            for item in items
            if (item["suite"], item["name"]) not in done_keys
        ]

    print(
        f"[tasks-eval] manifest={manifest_path} items={len(items)} "
        f"start={args.start_index} resume={args.resume}",
        flush=True,
    )
    client = ZMQPolicyClient(args.host, args.port)
    print(f"[tasks-eval] ping={client.ping()}", flush=True)
    suite_cache = {}
    settle_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)

    try:
        for index, item in enumerate(items, start=1):
            suite_name = item["suite"]
            if suite_name not in suite_cache:
                suite_cache[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
            suite_obj = suite_cache[suite_name]
            task = suite_obj.get_task(int(item["benchmark_index"]))
            if task.name != item["name"]:
                raise RuntimeError(
                    f"task order mismatch: {suite_name}[{item['benchmark_index']}]="
                    f"{task.name!r}, manifest={item['name']!r}"
                )
            init_states = suite_obj.get_task_init_states(int(item["benchmark_index"]))
            bddl_path = _resolve_bddl(task)
            max_steps = int(item.get("max_steps", 600))
            started = time.time()
            row = {
                "suite": suite_name,
                "base_task_index": item.get("base_task_index"),
                "base_task": item.get("base_task"),
                "category": item.get("category"),
                "difficulty_level": item.get("difficulty_level"),
                "name": item["name"],
                "benchmark_index": item["benchmark_index"],
                "max_steps": max_steps,
                "steps": 0,
                "success": False,
                "done": False,
                "elapsed_s": 0.0,
                "error": None,
            }
            env = None
            try:
                env = OffScreenRenderEnv(
                    bddl_file_name=str(bddl_path),
                    camera_heights=args.camera_size,
                    camera_widths=args.camera_size,
                )
                obs = env.reset()
                if len(init_states) > 0:
                    obs = env.set_init_state(init_states[0])
                for _ in range(args.settle_steps):
                    obs, _, _, _ = env.step(settle_action)

                actions = []
                for step in range(max_steps):
                    policy_obs = _rotate_180_if_requested(obs, args.rotate_180)
                    dial_obs = libero_obs_to_dial(
                        policy_obs, task.language, add_time_dim=True
                    )
                    dial_obs["annotation.human.action.task_description"] = [task.language]
                    action_dict = client.get_action(dial_obs)
                    action_chunk = dial_action_chunk_to_libero(
                        action_dict, n_action_steps=1
                    )
                    # LIBERO OSC expects [-1, 1] deltas; clip to match the
                    # official native-action client.
                    action = np.clip(action_chunk[0], -1.0, 1.0).astype(np.float32)
                    actions.append(action)
                    obs, reward, done, info = env.step(action)
                    success = (
                        bool(info.get("success", False)) if isinstance(info, dict) else False
                    )
                    row["steps"] = step + 1
                    row["done"] = bool(done)
                    row["success"] = bool(success or done)
                    if done or success:
                        break
                if actions:
                    arr = np.stack(actions, axis=0)
                    row["action_min"] = arr.min(axis=0).tolist()
                    row["action_max"] = arr.max(axis=0).tolist()
                    if args.save_actions:
                        np.save(result_dir / f"actions_{index:04d}.npy", arr)
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[tasks-eval] ERROR {item['name']}: {row['error']}", flush=True)
            finally:
                if env is not None:
                    env.close()
                row["elapsed_s"] = round(time.time() - started, 3)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"[tasks-eval] {index}/{len(items)} suite={suite_name} "
                    f"cat={row['category']} success={row['success']} "
                    f"steps={row['steps']}/{max_steps} elapsed={row['elapsed_s']}s "
                    f"name={item['name']}",
                    flush=True,
                )
    finally:
        client.close()

    all_rows = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    all_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    summary = _summary(all_rows)
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[tasks-eval] results={results_path}", flush=True)


if __name__ == "__main__":
    main()
