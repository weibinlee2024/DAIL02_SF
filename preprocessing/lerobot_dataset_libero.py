"""Pure pandas/imageio LeRobot v2.0 writer for LIBERO (no lerobot package required)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Protocol

import imageio.v2 as imageio
import numpy as np
import pandas as pd

STATE_DIM = 8
ACTION_DIM = 7

VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)

CHUNKS_SIZE = 1000

LIBERO_MODALITY = {
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
        "image": {"original_key": "observation.images.image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


class EpisodeLike(Protocol):
    state: np.ndarray
    action: np.ndarray
    agentview: np.ndarray
    wrist: np.ndarray
    language: str

    @property
    def num_frames(self) -> int: ...


def build_info_json(
    fps: float,
    image_size: int,
    total_episodes: int,
    total_frames: int,
    robot_type: str = "libero_franka",
) -> dict:
    return {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 0,
        "total_videos": total_episodes * len(VIDEO_KEYS),
        "total_chunks": max(1, (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE),
        "chunks_size": CHUNKS_SIZE,
        "fps": float(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": [
                    "eef_pos_x",
                    "eef_pos_y",
                    "eef_pos_z",
                    "eef_ori_x",
                    "eef_ori_y",
                    "eef_ori_z",
                    "gripper_qpos_0",
                    "gripper_qpos_1",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": [
                    "eef_delta_pos_x",
                    "eef_delta_pos_y",
                    "eef_delta_pos_z",
                    "eef_delta_ori_x",
                    "eef_delta_ori_y",
                    "eef_delta_ori_z",
                    "gripper",
                ],
            },
            "observation.images.image": {
                "dtype": "video",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": float(fps),
                    "video.height": image_size,
                    "video.width": image_size,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.wrist_image": {
                "dtype": "video",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": float(fps),
                    "video.height": image_size,
                    "video.width": image_size,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }


def encode_video_mp4(frames: np.ndarray, out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3), got {frames.shape}")

    try:
        writer = imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "23", "-preset", "fast"],
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return
    except Exception:
        if out_path.exists():
            out_path.unlink()

    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert proc.stdin is not None
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read() if proc.stderr else b""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(
            f"ffmpeg failed ({ret}): {stderr.decode('utf-8', errors='ignore')[:500]}"
        )


def episode_chunk(episode_index: int) -> int:
    return episode_index // CHUNKS_SIZE


def parquet_path(root: Path, episode_index: int) -> Path:
    chunk = episode_chunk(episode_index)
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def video_path(root: Path, episode_index: int, video_key: str) -> Path:
    chunk = episode_chunk(episode_index)
    return root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


class LiberoLeRobotWriter:
    """Incremental LeRobot v2.0 writer for LIBERO / RND dense episodes."""

    def __init__(
        self,
        root: Path,
        fps: float = 20.0,
        image_size: int = 128,
        robot_type: str = "libero_franka",
        resume: bool = False,
    ):
        self.root = Path(root)
        self.fps = float(fps)
        self.image_size = int(image_size)
        self.robot_type = robot_type
        self.resume = resume

        self.meta_dir = self.root / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.task2index: dict[str, int] = {}
        self.episodes: list[dict] = []
        self.total_frames = 0
        self.next_global_index = 0
        self.next_episode_index = 0

        if resume and (self.meta_dir / "episodes.jsonl").exists():
            self._load_resume_state()

    def _load_resume_state(self) -> None:
        tasks_path = self.meta_dir / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    row = json.loads(line)
                    self.task2index[row["task"]] = int(row["task_index"])

        with open(self.meta_dir / "episodes.jsonl") as f:
            for line in f:
                ep = json.loads(line)
                self.episodes.append(ep)
                self.total_frames += int(ep["length"])
                self.next_episode_index = max(
                    self.next_episode_index, int(ep["episode_index"]) + 1
                )

        self.next_global_index = self.total_frames

    def get_or_create_task_index(self, language: str) -> int:
        language = language.strip()
        if language not in self.task2index:
            self.task2index[language] = len(self.task2index)
        return self.task2index[language]

    def episode_exists(self, episode_index: int) -> bool:
        pq = parquet_path(self.root, episode_index)
        v0 = video_path(self.root, episode_index, VIDEO_KEYS[0])
        v1 = video_path(self.root, episode_index, VIDEO_KEYS[1])
        return pq.exists() and v0.exists() and v1.exists()

    def save_episode(
        self,
        episode: EpisodeLike,
        episode_index: Optional[int] = None,
        success: Optional[bool] = None,
    ) -> int:
        if episode_index is None:
            episode_index = self.next_episode_index

        if self.resume and self.episode_exists(episode_index):
            if not any(e["episode_index"] == episode_index for e in self.episodes):
                df = pd.read_parquet(parquet_path(self.root, episode_index))
                length = len(df)
                language = episode.language
                ep_record = {
                    "episode_index": episode_index,
                    "tasks": [language],
                    "length": length,
                }
                if success is not None:
                    ep_record["success"] = bool(success)
                self.episodes.append(ep_record)
                self.total_frames += length
                self.next_global_index = self.total_frames
            self.next_episode_index = max(self.next_episode_index, episode_index + 1)
            return episode_index

        t = episode.num_frames
        if episode.agentview.shape[1] != self.image_size:
            # Keep writer image_size as declared; do not silently rescale here.
            pass

        task_index = self.get_or_create_task_index(episode.language)
        frame_index = np.arange(t, dtype=np.int64)
        timestamps = frame_index.astype(np.float32) / self.fps
        global_index = np.arange(
            self.next_global_index, self.next_global_index + t, dtype=np.int64
        )

        df = pd.DataFrame(
            {
                "observation.state": list(episode.state.astype(np.float32)),
                "action": list(episode.action.astype(np.float32)),
                "timestamp": timestamps,
                "frame_index": frame_index,
                "episode_index": np.full(t, episode_index, dtype=np.int64),
                "index": global_index,
                "task_index": np.full(t, task_index, dtype=np.int64),
            }
        )

        pq = parquet_path(self.root, episode_index)
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)

        encode_video_mp4(
            episode.agentview, video_path(self.root, episode_index, VIDEO_KEYS[0]), self.fps
        )
        encode_video_mp4(
            episode.wrist, video_path(self.root, episode_index, VIDEO_KEYS[1]), self.fps
        )

        ep_record = {
            "episode_index": episode_index,
            "tasks": [episode.language],
            "length": t,
        }
        if success is not None:
            ep_record["success"] = bool(success)
        self.episodes.append(ep_record)
        self.total_frames += t
        self.next_global_index += t
        self.next_episode_index = max(self.next_episode_index, episode_index + 1)
        return episode_index

    def calculate_stats(self) -> dict:
        parquet_files = sorted(self.root.glob("data/chunk-*/episode_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files under {self.root}/data")

        all_states, all_actions = [], []
        for pq in parquet_files:
            df = pd.read_parquet(pq, columns=["observation.state", "action"])
            all_states.append(np.stack(df["observation.state"].to_numpy()))
            all_actions.append(np.stack(df["action"].to_numpy()))

        states = np.concatenate(all_states, axis=0).astype(np.float64)
        actions = np.concatenate(all_actions, axis=0).astype(np.float64)

        def _stats(arr: np.ndarray) -> dict:
            return {
                "mean": np.mean(arr, axis=0).tolist(),
                "std": np.std(arr, axis=0).tolist(),
                "min": np.min(arr, axis=0).tolist(),
                "max": np.max(arr, axis=0).tolist(),
                "q01": np.quantile(arr, 0.01, axis=0).tolist(),
                "q99": np.quantile(arr, 0.99, axis=0).tolist(),
            }

        return {
            "observation.state": _stats(states),
            "action": _stats(actions),
        }

    def finalize(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.episodes = sorted(self.episodes, key=lambda e: e["episode_index"])
        total_episodes = len(self.episodes)
        total_frames = sum(e["length"] for e in self.episodes)

        with open(self.meta_dir / "modality.json", "w") as f:
            json.dump(LIBERO_MODALITY, f, indent=4)

        info = build_info_json(
            fps=self.fps,
            image_size=self.image_size,
            total_episodes=total_episodes,
            total_frames=total_frames,
            robot_type=self.robot_type,
        )
        info["total_tasks"] = len(self.task2index)
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        with open(self.meta_dir / "episodes.jsonl", "w") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + "\n")

        tasks_sorted = sorted(self.task2index.items(), key=lambda kv: kv[1])
        with open(self.meta_dir / "tasks.jsonl", "w") as f:
            for task, task_index in tasks_sorted:
                f.write(json.dumps({"task_index": task_index, "task": task}) + "\n")

        stats = self.calculate_stats()
        with open(self.meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        n_succ = sum(1 for e in self.episodes if e.get("success") is True)
        n_fail = sum(1 for e in self.episodes if e.get("success") is False)
        print(
            f"[LiberoLeRobotWriter] finalized {self.root}: "
            f"{total_episodes} episodes ({n_succ} success / {n_fail} fail / "
            f"{total_episodes - n_succ - n_fail} unlabeled), "
            f"{total_frames} frames, {len(self.task2index)} tasks"
        )


def reset_output_dir(root: Path) -> None:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
