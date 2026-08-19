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


import argparse
import os
import sys
sys.path.append("/workspace/RLinf/denserwd")

import json
import numpy as np
from tqdm import tqdm
from hydra import compose, initialize
from omegaconf.omegaconf import OmegaConf

# from rlinf.config import validate_cfg
from denserwd.models import OctoReward
from denserwd.envs.libero_rm import LiberoEnvRM
from denserwd.utils.data_utils import save_dict_to_hdf5


SAVE_DATA_DIR = 'denserwd/data'
CONFIG_DIR = "../config/vla"


def main(args):
    initialize(config_path=CONFIG_DIR)
    cfg = compose(config_name=args.config_name)
    # cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    reward_model_cfg = cfg.reward_model
    reward_model_cfg.model.octo_pretrained_path = None
    use_reward_model = reward_model_cfg.use_reward_model
    reward_model = eval(reward_model_cfg.model.model_type)(reward_model_cfg).to(reward_model_cfg.device)
    reward_model_info = {
        "model": reward_model,
        "model_type": reward_model_cfg.model.model_type,
        "context_window": reward_model_cfg.model.context_window,
        "device": reward_model_cfg.device
    }
    num_envs = 2
    cfg.env.eval.auto_reset = True
    env = LiberoEnvRM(
        cfg.env.eval,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=2,
        worker_info=None,
        use_reward_model=cfg.reward_model.use_reward_model,
        reward_model_info=reward_model_info
    )

    libero_num_tasks = 2
    libero_num_trials_per_task = 2
    num_data_episodes = 1
    max_steps_per_episode = 20
    trail_ids_per_task = np.arange(libero_num_trials_per_task).tolist()
    num_data_episodes_per_task = num_data_episodes // libero_num_tasks
    
    np.random.seed(cfg.env.eval.seed)
    np.random.shuffle(trail_ids_per_task)

    total_cnt, step = 0, 0
    task_id, trail_pivot = 0, 0
    trail_id = trail_ids_per_task[trail_pivot]

    obs, _ = env.reset(reset_state_ids=[task_id * libero_num_trials_per_task + trail_id])
    success_cnt_per_task = [0 for _ in range(libero_num_tasks)]
    cnt_per_task = [0 for _ in range(libero_num_tasks)]
    progress_bar = tqdm(range(num_data_episodes), desc="Testing:")

    while total_cnt < num_data_episodes:
        if step == 0:
            print("\nCollecting data for task_id:", task_id, "trail_id:", trail_id)
            print("Task description:", obs["task_descriptions"])
        print("Step:", step, end="\r")

        action_chunk = np.ones((num_envs, 2, 7)) * np.random.rand()
        chunk_size = action_chunk.shape[1]
        next_obs, reward, done, _, info = env.chunk_step(action_chunk)

        step += chunk_size
        if done.any() or step >= max_steps_per_episode:
            # Update stats
            total_cnt += num_envs
            step = 0
            print()

            progress_bar.update(num_envs)

            trail_pivot += 1
            if trail_pivot >= num_data_episodes_per_task:
                trail_pivot = 0
                task_id = (task_id + 1) % libero_num_tasks
            trail_id = trail_ids_per_task[trail_pivot]
            # obs, _ = env.reset(reset_state_ids=[task_id * libero_num_trials_per_task + trail_id])

            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training reward model on LIBERO Benchmark")
    parser.add_argument("--config_name", type=str, help="Configuration file name", default="libero_spatial_openpi_rm")  
    parser.add_argument("--cuda_list", type=str, help="Cuda device list", default="7")  
    args = parser.parse_args()
    # args.config_name = "libero_spatial_collect_openpi"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_list
    main(args)   

