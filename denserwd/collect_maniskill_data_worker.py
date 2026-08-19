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


import copy
import os
import pickle as pkl
import json

import hydra
import numpy as np
import torch
from tqdm import tqdm
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker
from rlinf.utils.placement import HybridComponentPlacement
from denserwd.workers.huggingface_rollout_collector import MultiStepRolloutCollectorWorker


def extract_obs(obs, cfg):
    if not cfg.runner.record_task_description:
        obs.pop("task_descriptions", None)
    ret_obs = {}
    for key in obs:
        if "images" in key:
            ret_obs[key] = obs[key].clone()[0].float() / 255.0
        else:
            ret_obs[key] = obs[key][0]
    return ret_obs
    

@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_mlp_collect"
)
def main(cfg):
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout = MultiStepRolloutCollectorWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    rollout.init_worker()

    num_data_episodes = cfg.runner.num_data_episodes
    total_cnt = 0
    cfg.env.eval.init_params["control_mode"] = "pd_joint_delta_pos"
    env = ManiskillEnv(
        cfg.env.eval,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )

    data_list = []
    obs, _ = env.reset()
    success_cnt = 0
    progress_bar = tqdm(
        range(num_data_episodes), desc="Collecting Data Episodes:"
    )
    while success_cnt < num_data_episodes:
        action_handle = rollout.sample_actions(obs)
        action = action_handle.wait()
        action = action[0].squeeze(0)
        next_obs, reward, done, _, info = env.step(action)

        # Handle vector env
        single_obs = extract_obs(obs, cfg)
        single_next_obs = extract_obs(next_obs, cfg)
        single_action = action[0]
        single_reward = reward[0]
        single_done = done[0]

        # Handle chunk
        chunk_done = single_done[None, ...]
        chunk_reward = single_reward[None, ...]

        transition = copy.deepcopy(
            {
                "obs": single_obs,
                "next_obs": single_next_obs,
            }
        )
        data = copy.deepcopy(
            {
                "transitions": transition,
                "action": single_action,
                "rewards": chunk_reward,
                "dones": chunk_done,
                "terminations": chunk_done,
                "truncations": torch.zeros_like(chunk_done),
            }
        )
        data_list.append(data)

        obs = next_obs

        if done:
            success_cnt += reward
            total_cnt += 1
            print(
                f"{reward}\tGot {success_cnt} successes of {total_cnt} trials. {num_data_episodes} successes needed."
            )
            obs, _ = env.reset()
            progress_bar.update(1)

    save_file_path = os.path.join(cfg.runner.logger.log_path, "data.pkl")
    with open(save_file_path, "wb") as f:
        pkl.dump(data_list, f)
        print(
            f"Saved {num_data_episodes} demos with {len(data_list)} samples to {save_file_path}"
        )

    env.close()


if __name__ == "__main__":
    main()
