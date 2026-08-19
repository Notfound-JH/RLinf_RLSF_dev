# Override the default lerobot_dataset to fix bugs for few-shot learning
import torch
from itertools import accumulate
from pathlib import Path
from typing import Callable
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset


def get_episode_data_index(
    episode_dicts: dict[dict], episodes: list[int] | None = None
) -> dict[str, torch.Tensor]:
    episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()}
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}

    cumulative_lengths = list(accumulate(episode_lengths.values()))
    episode_data_index = {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }
    ep_idx_to_data_idx = {ep_idx: idx for idx, ep_idx in enumerate(episode_lengths.keys())}
    return episode_data_index, ep_idx_to_data_idx


class LeRobotDataset(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id,
            root,
            episodes,
            image_transforms,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync,
            download_videos,
            video_backend,
        )
        self.episode_data_index, self.ep_idx_to_data_idx = get_episode_data_index(self.meta.episodes, episodes)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        data_idx = self.ep_idx_to_data_idx[ep_idx]
        ep_start = self.episode_data_index["from"][data_idx]
        ep_end = self.episode_data_index["to"][data_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding


class LeRobotDatasetRM(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        context_window=5,
        sample_step=1,
        add_reverse=True,
    ):
        delta_timestamps = None  # we don't need it for reward model.
        super().__init__(
            repo_id,
            root,
            episodes,
            image_transforms,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync,
            download_videos,
            video_backend,
        )
        self.episode_data_index, self.ep_idx_to_data_idx = get_episode_data_index(self.meta.episodes, episodes)
        self.context_window = context_window
        self.sample_step = sample_step
        self.add_reverse = add_reverse  # whether to add reverse samples for VROC calculation
        self.task_embs = None

    def _forward_pad_and_concat(
        self,
        data: dict,
        current_length: int,
        target_length: int,
        skip_keys: set[str] | None = None,
    ) -> dict:
        skip_keys = skip_keys or set()
        new_data = {}
        for key in data.keys():
            if key in skip_keys:
                continue
            pad_values = [torch.zeros_like(data[key][0]) for _ in range(target_length - current_length)]
            new_data[key] = torch.stack(pad_values + data[key], dim=0)
        return new_data

    def _build_sequence(
        self,
        start_idx: int,
        end_idx: int,
        ep_start: int,
        ep_end: int,
        target_length: int,
        item: dict,
        skip_keys: set[str],
    ) -> tuple[dict, int]:
        start_idx = max(start_idx, ep_start)
        end_idx = min(end_idx, ep_end)

        data = self.hf_dataset[start_idx:end_idx]
        current_length = end_idx - start_idx
        data = self._forward_pad_and_concat(
            data,
            current_length=current_length,
            target_length=target_length,
            skip_keys=skip_keys,
        )
        data["timesteps"] = torch.cat(
            [
                torch.zeros(target_length - current_length),
                torch.arange(start_idx - ep_start + 1, end_idx - ep_start + 1),
            ]
        ).long()
        data["frame_index"] = item["frame_index"]
        data["episode_index"] = item["episode_index"]
        data["task_index"] = item["task_index"]
        if self.task_embs is not None:
            data["task_emb"] = self.task_embs[data["task_index"]]
        return data, current_length

    def _build_reversed_sequence(
        self,
        idx: int,
        ep_start: int,
        ep_end: int,
        ep_length: int,
        item: dict,
        skip_keys: set[str],
    ) -> dict:
        start_idx_rev = min(idx + self.context_window - 1, ep_end - 1)
        end_idx_rev = max(idx - 1, ep_start - 1)

        if end_idx_rev >= 0:
            data_rev = self.hf_dataset[start_idx_rev:end_idx_rev:-1]
        else:
            data_rev = self.hf_dataset[start_idx_rev::-1]
        current_length_rev = start_idx_rev - end_idx_rev
        data_rev = self._forward_pad_and_concat(
            data_rev,
            current_length=current_length_rev,
            target_length=self.context_window,
            skip_keys=skip_keys,
        )
        data_rev["timesteps"] = torch.cat(
            [
                torch.zeros(self.context_window - current_length_rev),
                torch.arange(ep_end - start_idx_rev, ep_end - end_idx_rev),
            ]
        ).long()
        data_rev["frame_index"] = ep_length - item["frame_index"] - 1
        data_rev["episode_index"] = item["episode_index"]
        data_rev["task_index"] = item["task_index"]
        if self.task_embs is not None:
            data_rev["task_emb"] = self.task_embs[data_rev["task_index"]]
        return data_rev

    def __len__(self):
        return self.num_frames // self.sample_step

    def __getitem__(self, idx) -> dict:
        idx *= self.sample_step
        idx = min(idx, self.num_frames - 1)

        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        ep_data_idx = self.ep_idx_to_data_idx[ep_idx]

        ep_start = self.episode_data_index["from"][ep_data_idx].item()
        ep_end = self.episode_data_index["to"][ep_data_idx].item()
        ep_length = ep_end - ep_start
        skip_keys = {"episode_index", "timestamp", "frame_index", "index", "task_index"}

        data, _ = self._build_sequence(
            start_idx=idx - self.context_window + 1,
            end_idx=idx + 1,
            ep_start=ep_start,
            ep_end=ep_end,
            target_length=self.context_window,
            item=item,
            skip_keys=skip_keys,
        )

        if self.add_reverse:
            data["reversed_samples"] = self._build_reversed_sequence(
                idx=idx,
                ep_start=ep_start,
                ep_end=ep_end,
                ep_length=ep_length,
                item=item,
                skip_keys=skip_keys,
            )
        return data

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


class LeRobotDatasetForTrainingRM(LeRobotDatasetRM):
    """Reward-model training dataset with both current and next-step windows."""

    def __getitem__(self, idx) -> dict:
        idx *= self.sample_step
        idx = min(idx, self.num_frames - 1)

        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        ep_data_idx = self.ep_idx_to_data_idx[ep_idx]

        ep_start = self.episode_data_index["from"][ep_data_idx].item()
        ep_end = self.episode_data_index["to"][ep_data_idx].item()
        ep_length = ep_end - ep_start
        skip_keys = {"episode_index", "timestamp", "frame_index", "index", "task_index"}

        data, _ = self._build_sequence(
            start_idx=idx - self.context_window + 1,
            end_idx=idx + 1,
            ep_start=ep_start,
            ep_end=ep_end,
            target_length=self.context_window,
            item=item,
            skip_keys=skip_keys,
        )

        next_item_idx = min(idx + 1, ep_end - 1)
        next_item = self.hf_dataset[next_item_idx]
        target_data, _ = self._build_sequence(
            start_idx=idx - self.context_window + 1,
            end_idx=idx + 2,
            ep_start=ep_start,
            ep_end=ep_end,
            target_length=self.context_window + 1,
            item=next_item,
            skip_keys=skip_keys,
        )
        data["target_model"] = target_data

        if self.add_reverse:
            data["reversed_samples"] = self._build_reversed_sequence(
                idx=idx,
                ep_start=ep_start,
                ep_end=ep_end,
                ep_length=ep_length,
                item=item,
                skip_keys=skip_keys,
            )
        return data
