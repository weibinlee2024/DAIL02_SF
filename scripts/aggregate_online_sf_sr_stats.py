#!/usr/bin/env python3
"""Aggregate online SF 3-arm SR across checkpoints / tags → mean±std.

Protocol fields always emitted:
  - n_episodes_per_task (from summary or dirname)
  - n_tasks
  - per-task rates when available

Example:
  python3 scripts/aggregate_online_sf_sr_stats.py \\
    --spec scripts/eval_protocol_ablation_spec_20261016.json \\
    --out /media/.../queue_eval_protocol_20261016/aggregate_mean_std.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _mean_std(xs: list[float]) -> dict[str, Any]:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "values": []}
    mu = sum(xs) / n
    if n == 1:
        return {"n": 1, "mean": mu, "std": 0.0, "values": xs}
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return {"n": n, "mean": mu, "std": math.sqrt(var), "values": xs}


def _fmt_pct(stat: dict[str, Any]) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    m, s, n = stat["mean"], stat["std"], stat["n"]
    return f"{100 * m:.1f}% ± {100 * (s or 0):.1f}% (n={n})"


def load_arm_from_results(results_json: Path) -> dict[str, Any] | None:
    if not results_json.is_file():
        return None
    d = json.loads(results_json.read_text())
    avg = d.get("average_success_rate", {})
    tasks = d.get("tasks") or {}
    out: dict[str, Any] = {
        "overall": (avg.get("overall") if isinstance(avg, dict) else avg),
        "PnPOnly": avg.get("PnPOnly") if isinstance(avg, dict) else None,
        "n_tasks": len(tasks),
        "per_task": {k: float(v) for k, v in tasks.items()} if tasks else {},
        "results_json": str(results_json),
    }
    return out


def load_run(ckpt: str, tag: str | None, eval_type: str, n_eps: int) -> dict[str, Any]:
    ckpt_p = Path(ckpt)
    # Prefer tagged summary used by queues; fall back to untagged.
    candidates = []
    if tag:
        candidates.append(ckpt_p / f"online_sf_sr_{eval_type}_{n_eps}eps_{tag}" / "summary_online_sf_sr.json")
    candidates.append(ckpt_p / f"online_sf_sr_{eval_type}_{n_eps}eps" / "summary_online_sf_sr.json")

    summary = None
    summary_path = None
    for c in candidates:
        if c.is_file():
            summary = json.loads(c.read_text())
            summary_path = str(c)
            break

    prefix = {
        "id": f"evaluation_sim_id_{n_eps}eps",
        "id6": f"evaluation_sim_id6_{n_eps}eps",
    }.get(eval_type, f"evaluation_sim_{eval_type}_{n_eps}eps")

    arms: dict[str, Any] = {}
    for mode in ("correct", "zero", "swap"):
        arm = None
        if summary and isinstance(summary.get("arms"), dict):
            a = summary["arms"].get(mode) or {}
            rj = Path(a.get("results_json") or "")
            arm = load_arm_from_results(rj) if rj else None
            if arm is None and a.get("overall") is not None:
                arm = {
                    "overall": a.get("overall"),
                    "PnPOnly": a.get("PnPOnly"),
                    "n_tasks": a.get("n_tasks"),
                    "per_task": {},
                    "results_json": a.get("results_json"),
                }
        if arm is None:
            arm = load_arm_from_results(ckpt_p / f"{prefix}_sf_{mode}" / "results.json")
        arms[mode] = arm

    n_from_summary = (summary or {}).get("n_episodes")
    return {
        "ckpt": str(ckpt_p),
        "tag": tag,
        "eval_type": eval_type,
        "n_episodes_per_task": int(n_from_summary or n_eps),
        "summary_path": summary_path,
        "arms": arms,
        "complete": all(arms.get(m) and arms[m].get("overall") is not None for m in ("correct", "zero", "swap")),
    }


def aggregate_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in runs if r.get("complete")]
    out: dict[str, Any] = {
        "n_runs_requested": len(runs),
        "n_runs_complete": len(done),
        "n_episodes_per_task": (done[0]["n_episodes_per_task"] if done else None),
        "protocol_note": "N_EPISODES is per-task; id6×10 → 60 episodes/arm total",
        "overall": {},
        "delta_vs_correct_pp": {},
        "per_task_correct": {},
    }
    for mode in ("correct", "zero", "swap"):
        vals = [r["arms"][mode]["overall"] for r in done if r["arms"].get(mode)]
        out["overall"][mode] = _mean_std(vals)
        out["overall"][mode]["mean_pct_str"] = _fmt_pct(out["overall"][mode])

    base_list = []
    for r in done:
        b = r["arms"]["correct"]["overall"]
        base_list.append(b)
        for mode in ("zero", "swap"):
            o = r["arms"][mode]["overall"]
            # stash deltas per run later
    for mode in ("zero", "swap"):
        deltas = []
        for r in done:
            b = r["arms"]["correct"]["overall"]
            o = r["arms"][mode]["overall"]
            if b is not None and o is not None:
                deltas.append((o - b) * 100.0)
        st = _mean_std(deltas)
        out["delta_vs_correct_pp"][mode] = {
            **st,
            "mean_pp_str": (
                f"{st['mean']:+.2f} ± {st['std']:.2f} pp (n={st['n']})" if st["mean"] is not None else "n/a"
            ),
        }

    # per-task on correct arm
    task_names: set[str] = set()
    for r in done:
        task_names |= set((r["arms"]["correct"].get("per_task") or {}).keys())
    for t in sorted(task_names):
        vals = []
        for r in done:
            pt = r["arms"]["correct"].get("per_task") or {}
            if t in pt:
                vals.append(pt[t])
        out["per_task_correct"][t] = {**_mean_std(vals), "mean_pct_str": _fmt_pct(_mean_std(vals))}

    out["runs"] = [
        {
            "ckpt": r["ckpt"],
            "tag": r["tag"],
            "complete": r["complete"],
            "correct": (r["arms"]["correct"] or {}).get("overall") if r["arms"].get("correct") else None,
            "zero": (r["arms"]["zero"] or {}).get("overall") if r["arms"].get("zero") else None,
            "swap": (r["arms"]["swap"] or {}).get("overall") if r["arms"].get("swap") else None,
        }
        for r in runs
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, required=True, help="JSON with comparison groups")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--eval-type", type=str, default="id6")
    ap.add_argument("--n-episodes", type=int, default=10)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    report: dict[str, Any] = {
        "eval_type": args.eval_type,
        "n_episodes_per_task": args.n_episodes,
        "total_episodes_per_arm_id6": 6 * args.n_episodes,
        "groups": {},
    }
    for gname, entries in spec.items():
        runs = []
        for e in entries:
            runs.append(
                load_run(
                    e["ckpt"],
                    e.get("tag"),
                    e.get("eval_type", args.eval_type),
                    int(e.get("n_episodes", args.n_episodes)),
                )
            )
        report["groups"][gname] = aggregate_group(runs)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:4000])
    print("WROTE", out_p)

    # human-readable table
    print("\n=== mean±std (overall SR %) ===")
    for gname, g in report["groups"].items():
        print(f"\n[{gname}] complete={g['n_runs_complete']}/{g['n_runs_requested']}  eps/task={g['n_episodes_per_task']}")
        for mode in ("correct", "zero", "swap"):
            print(f"  {mode:8s} {g['overall'][mode]['mean_pct_str']}")
        for mode in ("zero", "swap"):
            print(f"  Δ{mode:6s} {g['delta_vs_correct_pp'][mode]['mean_pp_str']}")


if __name__ == "__main__":
    main()
