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
import h5py
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from hydra import compose, initialize
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.envs.libero.libero_env import LiberoEnv
from denserwd.workers.huggingface_rollout_collector import MultiStepRolloutCollector
from denserwd.utils.data_utils import save_dict_to_hdf5


os.environ["EMBODIED_PATH"] = "/workspace/RLinf/examples/embodiment"

SAVE_DATA_DIR = 'data/datasets'
CONFIG_DIR = "./config/vla"


def extract_obs(obs):
    ret_obs, task = {}, None
    for key in obs:
        if "images" in key:
            ret_obs[key] = obs[key].clone().float().cpu().numpy() / 255.0
        elif key != "task_descriptions":
            ret_obs[key] = obs[key].cpu().numpy()
        else:
            task = obs[key][0]
    return ret_obs, task


class TrajectoryCollector:
    def __init__(self):
        self.traj_obs_main_images = []
        self.traj_obs_wrist_images = []
        self.traj_obs_robot_states = []
        self.traj_next_obs_main_images = []
        self.traj_next_obs_wrist_images = []
        self.traj_next_obs_robot_states = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_dones = []
    
    def add(self, obs, next_obs ,action, reward, done):
        self.traj_obs_main_images.append(obs["main_images"])
        self.traj_obs_wrist_images.append(obs["wrist_images"])
        self.traj_obs_robot_states.append(obs["states"])

        self.traj_next_obs_main_images.append(next_obs["main_images"])
        self.traj_next_obs_wrist_images.append(next_obs["wrist_images"])
        self.traj_next_obs_robot_states.append(next_obs["states"])

        self.traj_actions.append(action)
        self.traj_rewards.append(reward)
        self.traj_dones.append(done)
    
    def wrap_data(self):
        return {
            "obs_main_images": np.concatenate(self.traj_obs_main_images, axis=0),
            "obs_wrist_images": np.concatenate(self.traj_obs_wrist_images, axis=0),
            "obs_robot_states": np.concatenate(self.traj_obs_robot_states, axis=0),

            "next_obs_main_images": np.concatenate(self.traj_next_obs_main_images, axis=0),
            "next_obs_wrist_images": np.concatenate(self.traj_next_obs_wrist_images, axis=0),
            "next_obs_robot_states": np.concatenate(self.traj_next_obs_robot_states, axis=0),

            "actions": np.concatenate(self.traj_actions, axis=0),
            "rewards": np.concatenate(self.traj_rewards, axis=0),
            "dones": np.concatenate(self.traj_dones, axis=0),
        }
    
    def reset(self):
        for data_list in [self.traj_obs_main_images, self.traj_obs_wrist_images, self.traj_obs_robot_states,
                          self.traj_next_obs_main_images, self.traj_next_obs_wrist_images, self.traj_next_obs_robot_states,
                          self.traj_actions, self.traj_rewards, self.traj_dones]:
            data_list.clear()
    
    
def main(args):
    initialize(config_path=CONFIG_DIR)
    cfg = compose(config_name=args.config_name)
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    rollout = MultiStepRolloutCollector(cfg)

    env = LiberoEnv(
        cfg.env.eval,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    env.is_start = False

    libero_num_tasks = 10
    libero_num_trials_per_task = 50
    num_data_episodes = cfg.runner.num_data_episodes
    max_steps_per_episode = cfg.runner.max_steps_per_episode
    trail_ids_per_task = np.arange(libero_num_trials_per_task).tolist()
    num_data_episodes_per_task = num_data_episodes // libero_num_tasks
    
    np.random.seed(cfg.env.eval.seed)
    np.random.shuffle(trail_ids_per_task)

    save_file_dir = os.path.join(SAVE_DATA_DIR, args.config_name)
    os.makedirs(save_file_dir, exist_ok=True)

    total_cnt, step = 0, 0
    task_id, trail_pivot = 0, 0
    trail_id = trail_ids_per_task[trail_pivot]

    obs, _ = env.reset(reset_state_ids=[task_id * libero_num_trials_per_task + trail_id])
    success_cnt_per_task = [0 for _ in range(libero_num_tasks)]
    cnt_per_task = [0 for _ in range(libero_num_tasks)]
    
    # For streaming write
    current_h5_file = None
    current_task_name_for_file = None
    current_traj_index = 0

    traj_collector = TrajectoryCollector()
    num_data_episodes = num_data_episodes - task_id * num_data_episodes_per_task
    progress_bar = tqdm(range(num_data_episodes), desc="Collecting Data Episodes:")
    while total_cnt < num_data_episodes:
        if step == 0:
            print("\nCollecting data for task_id:", task_id, "trail_id:", trail_id)
            print("Task description:", obs["task_descriptions"])
            current_task_name = obs["task_descriptions"][0]
        print("Step:", step, end="\r")

        action_chunk = rollout.sample_actions(obs)
        chunk_size = action_chunk.shape[1]
        # Handle chunk
        for i in range(chunk_size):
            action = action_chunk[:, i, ...]
            next_obs, reward, done, _, info = env.step(action)

            # Handle vector env
            single_obs, task = extract_obs(obs)
            single_next_obs, _ = extract_obs(next_obs)
            traj_collector.add(
                obs=deepcopy(single_obs),
                next_obs=deepcopy(single_next_obs),
                action=deepcopy(action),
                reward=deepcopy(reward.cpu().numpy()),
                done=deepcopy(done.cpu().numpy()),
            )

            obs = next_obs

            step += 1

            if done or step >= max_steps_per_episode:
                # Update stats
                total_cnt += 1
                success_cnt_per_task[task_id] += int(done)
                cnt_per_task[task_id] += 1
                step = 0
                print()

                # Finalize trajectory
                traj_data = traj_collector.wrap_data()
                
                # Handle file switch if task changed
                if current_task_name_for_file != current_task_name:
                    if current_h5_file is not None:
                        current_h5_file.close()
                    clean_name = current_task_name.replace(" ", "_")
                    save_path = os.path.join(save_file_dir, f"{clean_name}_data.hdf5")
                    current_h5_file = h5py.File(save_path, "a")
                    current_task_name_for_file = current_task_name
                    current_traj_index = len(current_h5_file.keys())  # continue numbering
                    
                traj_collector.reset()

                # Save trajectory
                traj_key = f"traj_{current_traj_index}"
                save_dict_to_hdf5(traj_data, current_h5_file, traj_key)
                print(f"Saved trajectory {traj_key} for task '{current_task_name}' at {save_path}")
                current_traj_index += 1
                
                progress_bar.update(1)

                trail_pivot += 1
                if trail_pivot >= num_data_episodes_per_task:
                    trail_pivot = 0
                    task_id = (task_id + 1) % libero_num_tasks
                trail_id = trail_ids_per_task[trail_pivot]
                obs, _ = env.reset(reset_state_ids=[task_id * libero_num_trials_per_task + trail_id])

                break

    # Close final file
    if current_h5_file is not None:
        current_h5_file.close()

    print(f"Success rate: {sum(success_cnt_per_task)/num_data_episodes:.3f}")
    for t_id in range(libero_num_tasks):
        if cnt_per_task[t_id] > 0:
            print(f"  Task {t_id}: Success rate: {success_cnt_per_task[t_id]/cnt_per_task[t_id]:.3f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training reward model on LIBERO Benchmark")
    parser.add_argument("--config_name", type=str, help="Configuration file name", default="libero_40_collect_mlp")  
    parser.add_argument("--cuda_list", type=str, help="Cuda device list", default="7")  
    args = parser.parse_args()
    # args.config_name = "libero_spatial_collect_openvla"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_list
    main(args)   

