# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["HF_LEROBOT_HOME"] = "/workspace/public_dataset/"
import json
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader
from denserwd.utils.dataset_utils import create_data_loader, create_torch_dataset
from denserwd.datasets.lerobot_dataset import LeRobotDatasetRM


class FSDPSftWorker():
    def __init__(self, cfg: DictConfig):

        self.cfg = cfg
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.device = torch.cuda.current_device()

        self.data_loader, self.data_config = self.build_dataloader()
        self.data_iter = iter(self.data_loader)
        self.global_step = 0

    def build_dataloader(self):
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
        # breakpoint()
        config = get_openpi_config(
            self.cfg.actor.model.openpi.config_name,
            model_path=self.cfg.actor.model.model_path,
            batch_size=1,
        )
        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = create_torch_dataset(data_config, model_config=config.model, action_horizon=config.model.action_horizon, 
                                       episodes=[0, 1, 2], dataset_cls=LeRobotDatasetRM, context_window=5, sample_step=5)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        return data_loader, data_config

    def run_training(self):
        for idx, data in enumerate(self.data_loader):
            print(f"Step {idx}:", data['actions'])
            print(data['timesteps'])
            if idx >= 10:
                break

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

os.environ["EMBODIED_PATH"] = "/workspace/RLinf/denserwd"


@hydra.main(
    version_base="1.1", config_path="../config/vla", config_name="libero_sft_openpi"
)
def main(cfg) -> None:

    sft = FSDPSftWorker(cfg)
    sft.run_training()


if __name__ == "__main__":
    main()
