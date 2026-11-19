#!/usr/bin/env python3
"""Single-task LIBERO-plus eval smoke over the DIAL ZMQ policy server.

Runs in the ``libero-plus`` conda env (libero/mujoco/robosuite) and talks to
``scripts/inference_service_dial.py --server`` running in the ``dial`` env.

This is an interface smoke test, not a benchmark result: it verifies
env reset/settle, obs -> DialPolicy keys, 7D action conversion, env.step, and
the success/done plumbing. Use the official OpenWAM LIBERO-plus harness for
real numbers.

Example:
  LIBERO_CONFIG_PATH=~/.libero-openwam-plus \
  PYTHONPATH=/tmp/libero_eval_deps:/path/to/DIAL02_SF20161018_2 \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  /home/weibin02/miniconda3/envs/libero-plus/bin/python \
    scripts/eval_libero_plus_smoke.py \
      --port 5577 --suite libero_spatial --task-id 0 --max-steps 20
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# These are installed in /tmp/libero_eval_deps for the libero-plus env (the
# env itself does not have pyzmq/msgpack).
sys.path.insert(0, "/tmp/libero_eval_deps")

import msgpack  # noqa: E402
import zmq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr00t.eval.libero_obs_action import (  # noqa: E402
    dial_action_chunk_to_libero,
    libero_obs_to_dial,
)


def _encode_custom(obj):
    if isinstance(obj, np.ndarray):
        output = io.BytesIO()
        np.save(output, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    return obj


def _decode_custom(obj):
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def to_bytes(data: dict) -> bytes:
    return msgpack.packb(data, default=_encode_custom)


def from_bytes(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=_decode_custom, raw=False)


class ZMQPolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 120000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")
        self.api_token = None

    def call(self, endpoint: str, data=None, requires_input: bool = True):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token is not None:
            request["api_token"] = self.api_token
        self.socket.send(to_bytes(request))
        response = from_bytes(self.socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"policy server error: {response['error']}")
        return response

    def ping(self):
        return self.call("ping", requires_input=False)

    def get_action(self, obs: dict):
        return self.call("get_action", obs)

    def close(self):
        self.socket.close()
        self.context.term()


def _rotate_180_if_requested(obs: dict, enabled: bool) -> dict:
    if not enabled:
        return obs
    out = dict(obs)
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key in out and isinstance(out[key], np.ndarray):
            out[key] = out[key][::-1, ::-1]
    return out


def _resolve_bddl(task) -> Path:
    from libero.libero import get_libero_path

    bddl_files = Path(get_libero_path("bddl_files"))
    return bddl_files / task.problem_folder / task.bddl_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5577)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--rotate-180", action="store_true", default=True)
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false")
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--debug-dir", default="")
    args = parser.parse_args()

    # LIBERO config (assets/bddl/init states) is created by the OpenWAM setup.
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path.home() / ".libero-openwam-plus"))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    bddl_path = _resolve_bddl(task)
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL not found: {bddl_path}")

    print(f"[eval-smoke] suite={args.suite} task_id={args.task_id} task={task.name}")
    print(f"[eval-smoke] language={task.language!r}")
    print(f"[eval-smoke] bddl={bddl_path}")
    print(f"[eval-smoke] init_states={len(init_states)}")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    client = ZMQPolicyClient(args.host, args.port)
    ping = client.ping()
    print(f"[eval-smoke] server ping={ping}")

    settle_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    result = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task": task.name,
        "language": task.language,
        "trial": args.trial,
        "steps": 0,
        "success": False,
        "done": False,
        "action_min": None,
        "action_max": None,
        "error": None,
    }

    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        obs = env.reset()
        if len(init_states) > 0:
            obs = env.set_init_state(init_states[args.trial % len(init_states)])
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(settle_action)

        all_actions = []
        for step in range(args.max_steps):
            policy_obs = _rotate_180_if_requested(obs, args.rotate_180)
            dial_obs = libero_obs_to_dial(policy_obs, task.language, add_time_dim=True)
            # DialPolicy converts non-array values to np.array; a bare string
            # becomes a 0-D array and breaks the language transform. Use a
            # length-1 list, matching the server's example client.
            dial_obs["annotation.human.action.task_description"] = [task.language]
            action_dict = client.get_action(dial_obs)
            action_chunk = dial_action_chunk_to_libero(action_dict, n_action_steps=1)
            # LIBERO OSC expects [-1, 1] deltas. The DIAL head can emit
            # out-of-range values; match the official client's clipping.
            action = np.clip(action_chunk[0], -1.0, 1.0).astype(np.float32)
            all_actions.append(action)
            obs, reward, done, info = env.step(action)
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            result["steps"] = step + 1
            result["done"] = bool(done)
            result["success"] = bool(success or done)
            if debug_dir is not None:
                np.save(debug_dir / f"step_{step:04d}_action.npy", action)
            if done or success:
                break

        if all_actions:
            actions = np.stack(all_actions, axis=0)
            result["action_min"] = actions.min(axis=0).tolist()
            result["action_max"] = actions.max(axis=0).tolist()
        print(f"[eval-smoke] steps={result['steps']} done={result['done']} success={result['success']}")
        print(f"[eval-smoke] action keys={sorted(action_dict.keys())}")
        print(f"[eval-smoke] action shape={tuple(action.shape)}")
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[eval-smoke] ERROR {result['error']}", file=sys.stderr)
        raise
    finally:
        client.close()
        env.close()
        if args.result_dir:
            out_dir = Path(args.result_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "result.json").open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            print(f"[eval-smoke] result={out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
