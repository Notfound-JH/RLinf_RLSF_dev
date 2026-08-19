#!/usr/bin/env python3
"""Convert flat LIBERO HDF5 datasets into a single local LeRobot-style dataset.

This script reads source HDF5 files under ``data/datasets`` and writes a new
LeRobot-style dataset directory with the following layout:

- ``meta/info.json``
- ``meta/tasks.jsonl``
- ``meta/episodes.jsonl``
- ``meta/stats.json``
- ``data/chunk-XXX/episode_XXXXXX.parquet``

The source HDF5 files are never modified. Each ``traj_*`` group becomes one
episode parquet file.

By default, the script merges the four LIBERO benchmark directories below into
one LeRobot dataset when they exist under ``--source-dir``:

- ``libero_spatial_collect_openpi``
- ``libero_object_collect_openpi``
- ``libero_goal_collect_openpi``
- ``libero_10_collect_openpi``
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value
from tqdm import tqdm


DEFAULT_SOURCE_DIR = Path("/workspace/RLinf/denserwd/data/datasets")
DEFAULT_OUTPUT_DIR = Path("/workspace/RLinf/denserwd/data/lerobot_libero")
DEFAULT_DATASET_NAME = "libero_combined_openpi"
DEFAULT_FPS = 10
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_LIBERO_SUBDIRS = (
    "libero_spatial_collect_openpi",
    "libero_object_collect_openpi",
    "libero_goal_collect_openpi",
    "libero_10_collect_openpi",
)


@dataclass
class RunningStats:
    """Keep streaming stats for vector features."""

    sum: np.ndarray
    sum_sq: np.ndarray
    min: np.ndarray
    max: np.ndarray
    count: int = 0

    @classmethod
    def create(cls, dim: int) -> "RunningStats":
        return cls(
            sum=np.zeros(dim, dtype=np.float64),
            sum_sq=np.zeros(dim, dtype=np.float64),
            min=np.full(dim, np.inf, dtype=np.float64),
            max=np.full(dim, -np.inf, dtype=np.float64),
            count=0,
        )

    def update(self, values: np.ndarray) -> None:
        """Update stats with a batch of shape [N, D]."""
        if values.size == 0:
            return

        values = np.asarray(values, dtype=np.float64)
        self.sum += values.sum(axis=0)
        self.sum_sq += np.square(values).sum(axis=0)
        self.min = np.minimum(self.min, values.min(axis=0))
        self.max = np.maximum(self.max, values.max(axis=0))
        self.count += int(values.shape[0])

    def merge(self, other: "RunningStats") -> None:
        """Merge another RunningStats object into this one."""
        if other.count == 0:
            return
        if self.count == 0:
            self.sum = other.sum.copy()
            self.sum_sq = other.sum_sq.copy()
            self.min = other.min.copy()
            self.max = other.max.copy()
            self.count = other.count
            return

        self.sum += other.sum
        self.sum_sq += other.sum_sq
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)
        self.count += other.count

    def to_json(self) -> dict[str, Any]:
        """Serialize stats to a JSON-compatible dict."""
        if self.count == 0:
            raise ValueError("Cannot serialize empty stats.")

        mean = self.sum / self.count
        var = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        std = np.sqrt(var)
        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }


@dataclass
class EpisodeWriteJob:
    """One trajectory-to-parquet conversion job."""

    traj_key: str
    episode_index: int
    start_frame_index: int
    parquet_path: str


@dataclass
class FileWriteJob:
    """A group of episode write jobs for a single HDF5 file."""

    task_index: int
    task_name: str
    source_subdir: str
    hdf5_path: str
    episode_jobs: list[EpisodeWriteJob]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Convert flat HDF5 LIBERO datasets to a single LeRobot parquet dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Root directory containing source HDF5 subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the converted LeRobot dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Name stored in metadata for the converted dataset.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="panda",
        help="Robot type stored in meta/info.json.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames per second stored in metadata.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of episodes per parquet chunk directory.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1))),
        help="Number of worker processes used to convert HDF5 files in parallel.",
    )
    parser.add_argument(
        "--source-subdirs",
        nargs="+",
        default=None,
        help=(
            "Specific source subdirectories to merge into one LeRobot dataset. "
            "If omitted, the script uses the four standard LIBERO directories when present."
        ),
    )
    parser.add_argument(
        "--task-prefix",
        type=str,
        default=None,
        help="Optional prefix filter for subdirectory names, such as 'libero_10'.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for the number of converted episodes, useful for validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the source dataset and report the conversion plan without writing files.",
    )
    return parser.parse_args()


def normalize_task_name(file_path: Path) -> str:
    """Convert ``*_data.hdf5`` to a task name."""
    stem = file_path.stem
    suffix = "_data"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def sorted_traj_keys(hdf5_file: h5py.File) -> list[str]:
    """Return trajectory keys sorted by numeric suffix."""
    traj_keys = [key for key in hdf5_file.keys() if key.startswith("traj_")]
    return sorted(traj_keys, key=lambda key: int(key.split("_")[-1]))


def to_uint8_images(images: np.ndarray) -> np.ndarray:
    """Convert images to uint8 HWC arrays."""
    if images.dtype == np.uint8:
        return images

    images = np.asarray(images)
    if np.issubdtype(images.dtype, np.floating):
        if images.max() <= 1.0 + 1e-6:
            images = np.clip(np.rint(images * 255.0), 0, 255)
        else:
            images = np.clip(np.rint(images), 0, 255)
    return images.astype(np.uint8)


def build_features(state_dim: int, action_dim: int) -> Features:
    """Create Hugging Face dataset features for one episode parquet."""
    return Features(
        {
            "image": Image(),
            "wrist_image": Image(),
            "next_image": Image(),
            "next_wrist_image": Image(),
            "state": Sequence(Value("float32"), length=state_dim),
            "actions": Sequence(Value("float32"), length=action_dim),
            "next_state": Sequence(Value("float32"), length=state_dim),
            "timestamp": Value("float32"),
            "frame_index": Value("int64"),
            "episode_index": Value("int64"),
            "index": Value("int64"),
            "task_index": Value("int64"),
            "rewards": Value("float32"),
            "dones": Value("bool"),
        }
    )


def resolve_source_subdirs(
    source_dir: Path,
    source_subdirs: list[str] | None,
    task_prefix: str | None,
) -> list[Path]:
    """Resolve which immediate subdirectories under source_dir should be merged."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    all_subdirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if source_subdirs:
        resolved = []
        for subdir_name in source_subdirs:
            subdir_path = source_dir / subdir_name
            if not subdir_path.is_dir():
                raise FileNotFoundError(f"Source subdirectory does not exist: {subdir_path}")
            resolved.append(subdir_path)
        return resolved

    preferred = [source_dir / name for name in DEFAULT_LIBERO_SUBDIRS if (source_dir / name).is_dir()]
    if preferred:
        if task_prefix is not None:
            preferred = [path for path in preferred if path.name.startswith(task_prefix)]
        if preferred:
            return preferred

    if task_prefix is not None:
        all_subdirs = [path for path in all_subdirs if path.name.startswith(task_prefix)]

    if not all_subdirs:
        raise FileNotFoundError(
            f"No source subdirectories found under {source_dir} with task_prefix={task_prefix!r}."
        )
    return all_subdirs


def collect_hdf5_files(
    source_dir: Path,
    source_subdirs: list[str] | None,
    task_prefix: str | None,
) -> tuple[list[Path], list[str]]:
    """Collect source HDF5 files from the selected subdirectories."""
    selected_dirs = resolve_source_subdirs(source_dir, source_subdirs, task_prefix)
    files: list[Path] = []
    selected_names: list[str] = []

    for subdir in selected_dirs:
        subdir_files = sorted(subdir.glob("*.hdf5"))
        if not subdir_files:
            continue
        files.extend(subdir_files)
        selected_names.append(subdir.name)

    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found in selected subdirectories under {source_dir}."
        )
    return files, selected_names


def plan_conversion(hdf5_files: list[Path], max_episodes: int | None) -> dict[str, Any]:
    """Inspect source files and produce a conversion plan."""
    total_frames = 0
    total_episodes = 0
    task_records: list[dict[str, Any]] = []
    state_dim = None
    action_dim = None

    for task_index, hdf5_path in enumerate(hdf5_files):
        with h5py.File(hdf5_path, "r") as hdf5_file:
            traj_keys = sorted_traj_keys(hdf5_file)
            task_name = normalize_task_name(hdf5_path)
            task_records.append(
                {
                    "task_index": task_index,
                    "task": task_name,
                    "source_file": str(hdf5_path),
                    "source_subdir": hdf5_path.parent.name,
                    "num_episodes": len(traj_keys),
                }
            )

            for traj_key in traj_keys:
                if max_episodes is not None and total_episodes >= max_episodes:
                    break

                traj_group = hdf5_file[traj_key]
                episode_length = int(traj_group["actions"].shape[0])
                total_frames += episode_length
                total_episodes += 1

                if state_dim is None:
                    state_dim = int(traj_group["obs_robot_states"].shape[-1])
                if action_dim is None:
                    action_dim = int(traj_group["actions"].shape[-1])

    if state_dim is None or action_dim is None:
        raise ValueError("Failed to infer state/action dimensions from source data.")

    return {
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "tasks": task_records,
    }


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create the output directory if needed."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Use --overwrite to write into it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with indentation."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL records."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_episode_dataset(
    traj_group: h5py.Group,
    episode_index: int,
    task_index: int,
    start_frame_index: int,
    fps: int,
    features: Features,
) -> tuple[Dataset, int, dict[str, np.ndarray]]:
    """Convert one HDF5 trajectory group into a HF dataset episode."""
    main_images = to_uint8_images(traj_group["obs_main_images"][()])
    wrist_images = to_uint8_images(traj_group["obs_wrist_images"][()])
    next_main_images = to_uint8_images(traj_group["next_obs_main_images"][()])
    next_wrist_images = to_uint8_images(traj_group["next_obs_wrist_images"][()])
    states = np.asarray(traj_group["obs_robot_states"][()], dtype=np.float32)
    next_states = np.asarray(traj_group["next_obs_robot_states"][()], dtype=np.float32)
    actions = np.asarray(traj_group["actions"][()], dtype=np.float32)
    rewards = np.asarray(traj_group["rewards"][()], dtype=np.float32)
    dones = np.asarray(traj_group["dones"][()], dtype=bool)

    length = int(actions.shape[0])
    frame_index = np.arange(length, dtype=np.int64)
    index = start_frame_index + frame_index
    timestamp = frame_index.astype(np.float32) / float(fps)

    payload = {
        "image": list(main_images),
        "wrist_image": list(wrist_images),
        "next_image": list(next_main_images),
        "next_wrist_image": list(next_wrist_images),
        "state": [row for row in states],
        "actions": [row for row in actions],
        "next_state": [row for row in next_states],
        "timestamp": timestamp.tolist(),
        "frame_index": frame_index.tolist(),
        "episode_index": [episode_index] * length,
        "index": index.tolist(),
        "task_index": [task_index] * length,
        "rewards": rewards.tolist(),
        "dones": dones.tolist(),
    }
    return Dataset.from_dict(payload, features=features), length, {
        "state": states,
        "actions": actions,
        "next_state": next_states,
    }


def build_file_jobs(
    hdf5_files: list[Path],
    output_dir: Path,
    chunk_size: int,
    max_episodes: int | None,
) -> tuple[list[FileWriteJob], int, int]:
    """Precompute all file-level conversion jobs with stable output indices."""
    file_jobs: list[FileWriteJob] = []
    episode_index = 0
    total_frames = 0

    for task_index, hdf5_path in enumerate(hdf5_files):
        if max_episodes is not None and episode_index >= max_episodes:
            break

        episode_jobs: list[EpisodeWriteJob] = []
        with h5py.File(hdf5_path, "r") as hdf5_file:
            for traj_key in sorted_traj_keys(hdf5_file):
                if max_episodes is not None and episode_index >= max_episodes:
                    break

                traj_length = int(hdf5_file[traj_key]["actions"].shape[0])
                chunk_dir = output_dir / "data" / f"chunk-{episode_index // chunk_size:03d}"
                parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
                episode_jobs.append(
                    EpisodeWriteJob(
                        traj_key=traj_key,
                        episode_index=episode_index,
                        start_frame_index=total_frames,
                        parquet_path=str(parquet_path),
                    )
                )
                episode_index += 1
                total_frames += traj_length

        if episode_jobs:
            file_jobs.append(
                FileWriteJob(
                    task_index=task_index,
                    task_name=normalize_task_name(hdf5_path),
                    source_subdir=hdf5_path.parent.name,
                    hdf5_path=str(hdf5_path),
                    episode_jobs=episode_jobs,
                )
            )

    return file_jobs, episode_index, total_frames


def convert_file_job(
    file_job: FileWriteJob,
    fps: int,
    state_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    """Convert all trajectories from one HDF5 file into parquet episodes."""
    features = build_features(state_dim=state_dim, action_dim=action_dim)
    state_stats = RunningStats.create(state_dim)
    action_stats = RunningStats.create(action_dim)
    next_state_stats = RunningStats.create(state_dim)
    episodes_meta: list[dict[str, Any]] = []
    frames_written = 0

    with h5py.File(file_job.hdf5_path, "r") as hdf5_file:
        for episode_job in file_job.episode_jobs:
            traj_group = hdf5_file[episode_job.traj_key]
            dataset, episode_length, arrays = build_episode_dataset(
                traj_group=traj_group,
                episode_index=episode_job.episode_index,
                task_index=file_job.task_index,
                start_frame_index=episode_job.start_frame_index,
                fps=fps,
                features=features,
            )

            parquet_path = Path(episode_job.parquet_path)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(str(parquet_path))

            state_stats.update(arrays["state"])
            action_stats.update(arrays["actions"])
            next_state_stats.update(arrays["next_state"])
            frames_written += episode_length
            episodes_meta.append(
                {
                    "episode_index": episode_job.episode_index,
                    "task_index": file_job.task_index,
                    "task": file_job.task_name,
                    "source_subdir": file_job.source_subdir,
                    "length": episode_length,
                }
            )

    return {
        "episodes_meta": episodes_meta,
        "state_stats": state_stats,
        "action_stats": action_stats,
        "next_state_stats": next_state_stats,
        "frames_written": frames_written,
    }


def convert_dataset(args: argparse.Namespace) -> None:
    """Run the conversion."""
    hdf5_files, selected_subdirs = collect_hdf5_files(
        args.source_dir,
        args.source_subdirs,
        args.task_prefix,
    )
    plan = plan_conversion(hdf5_files, args.max_episodes)
    file_jobs, planned_episode_count, planned_frame_count = build_file_jobs(
        hdf5_files=hdf5_files,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        max_episodes=args.max_episodes,
    )
    num_workers = max(1, int(args.num_workers))
    if len(file_jobs) <= 1:
        num_workers = 1

    print("=" * 80)
    print("HDF5 -> LeRobot conversion plan")
    print("=" * 80)
    print(f"Source dir        : {args.source_dir}")
    print(f"Selected subdirs  : {', '.join(selected_subdirs)}")
    print(f"Output dir        : {args.output_dir}")
    print(f"Task files        : {len(hdf5_files)}")
    print(f"Planned file jobs : {len(file_jobs)}")
    print(f"Total tasks       : {len(plan['tasks'])}")
    print(f"Total episodes    : {planned_episode_count}")
    print(f"Total frames      : {planned_frame_count}")
    print(f"State dim         : {plan['state_dim']}")
    print(f"Action dim        : {plan['action_dim']}")
    print(f"Chunk size        : {args.chunk_size}")
    print(f"FPS               : {args.fps}")
    print(f"Worker processes  : {num_workers}")
    print(f"Dry run           : {args.dry_run}")
    print("=" * 80)

    if args.dry_run:
        return

    ensure_output_dir(args.output_dir, args.overwrite)

    state_stats = RunningStats.create(plan["state_dim"])
    action_stats = RunningStats.create(plan["action_dim"])
    next_state_stats = RunningStats.create(plan["state_dim"])
    episodes_meta: list[dict[str, Any]] = []
    tasks_meta = [
        {
            "task_index": task["task_index"],
            "task": task["task"],
            "source_subdir": task["source_subdir"],
        }
        for task in plan["tasks"]
    ]

    if num_workers == 1:
        results = [
            convert_file_job(
                file_job=file_job,
                fps=args.fps,
                state_dim=plan["state_dim"],
                action_dim=plan["action_dim"],
            )
            for file_job in tqdm(file_jobs, desc="Files")
        ]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        convert_file_job,
                        file_jobs,
                        [args.fps] * len(file_jobs),
                        [plan["state_dim"]] * len(file_jobs),
                        [plan["action_dim"]] * len(file_jobs),
                    ),
                    total=len(file_jobs),
                    desc="Files",
                )
            )

    total_frames_written = 0
    for result in results:
        episodes_meta.extend(result["episodes_meta"])
        state_stats.merge(result["state_stats"])
        action_stats.merge(result["action_stats"])
        next_state_stats.merge(result["next_state_stats"])
        total_frames_written += int(result["frames_written"])

    episodes_meta.sort(key=lambda row: row["episode_index"])

    info = {
        "codebase_version": "v2.0",
        "dataset_name": args.dataset_name,
        "robot_type": args.robot_type,
        "source_dir": str(args.source_dir),
        "source_subdirs": selected_subdirs,
        "total_episodes": planned_episode_count,
        "total_frames": total_frames_written,
        "total_tasks": len(tasks_meta),
        "total_videos": 0,
        "total_chunks": (planned_episode_count + args.chunk_size - 1) // args.chunk_size,
        "chunks_size": args.chunk_size,
        "fps": args.fps,
        "splits": {
            "train": f"0:{planned_episode_count}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {
                "dtype": "image",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "next_image": {
                "dtype": "image",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "next_wrist_image": {
                "dtype": "image",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": [plan["state_dim"]],
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": [plan["action_dim"]],
                "names": ["actions"],
            },
            "next_state": {
                "dtype": "float32",
                "shape": [plan["state_dim"]],
                "names": ["state"],
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "rewards": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "dones": {
                "dtype": "bool",
                "shape": [1],
                "names": None,
            },
        },
    }

    stats = {
        "state": state_stats.to_json(),
        "actions": action_stats.to_json(),
        "next_state": next_state_stats.to_json(),
    }

    write_json(args.output_dir / "meta" / "info.json", info)
    write_jsonl(args.output_dir / "meta" / "tasks.jsonl", tasks_meta)
    write_jsonl(args.output_dir / "meta" / "episodes.jsonl", episodes_meta)
    write_json(args.output_dir / "meta" / "stats.json", stats)

    print(f"Converted {planned_episode_count} episodes and {total_frames_written} frames.")
    print(f"Output written to: {args.output_dir}")


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    convert_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
