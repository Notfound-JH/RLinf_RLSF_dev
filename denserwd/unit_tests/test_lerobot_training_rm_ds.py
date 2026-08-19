import torch

from denserwd.datasets.lerobot_dataset import LeRobotDatasetForTrainingRM


class FakeHFDataset:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            selected = self.samples[idx]
            return {
                key: [sample[key] for sample in selected]
                for key in selected[0].keys()
            }
        return self.samples[idx]


def make_sample(value: int, episode_index: int = 0) -> dict:
    return {
        "observation": torch.tensor([value], dtype=torch.float32),
        "action": torch.tensor([value + 10], dtype=torch.float32),
        "episode_index": torch.tensor(episode_index),
        "frame_index": torch.tensor(value),
        "task_index": torch.tensor(0),
        "timestamp": torch.tensor(float(value)),
        "index": torch.tensor(value),
    }


def build_dataset(context_window: int = 3) -> LeRobotDatasetForTrainingRM:
    dataset = LeRobotDatasetForTrainingRM.__new__(LeRobotDatasetForTrainingRM)
    dataset.hf_dataset = FakeHFDataset([make_sample(i) for i in range(4)])
    dataset.episode_data_index = {
        "from": torch.LongTensor([0]),
        "to": torch.LongTensor([4]),
    }
    dataset.ep_idx_to_data_idx = {0: 0}
    dataset.context_window = context_window
    dataset.sample_step = 1
    dataset.add_reverse = False
    dataset.task_embs = None
    dataset.num_frames = 4
    return dataset


def test_training_rm_dataset_returns_current_and_target_windows():
    dataset = build_dataset(context_window=3)

    item = dataset[0]

    assert item["observation"].shape == (3, 1)
    assert torch.equal(item["timesteps"], torch.tensor([0, 0, 1]))
    assert torch.equal(item["observation"].squeeze(-1), torch.tensor([0.0, 0.0, 0.0]))

    target = item["target_model"]
    assert target["observation"].shape == (4, 1)
    assert torch.equal(target["timesteps"], torch.tensor([0, 0, 1, 2]))
    assert torch.equal(target["observation"].squeeze(-1), torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert target["frame_index"].item() == 1


def test_training_rm_dataset_clamps_target_window_at_episode_end():
    dataset = build_dataset(context_window=3)

    item = dataset[3]
    target = item["target_model"]

    assert torch.equal(item["timesteps"], torch.tensor([2, 3, 4]))
    assert torch.equal(target["timesteps"], torch.tensor([0, 2, 3, 4]))
    assert target["frame_index"].item() == 3
