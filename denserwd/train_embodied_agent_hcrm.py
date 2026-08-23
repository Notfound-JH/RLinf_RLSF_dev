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
import json
import os

import torch.multiprocessing as mp
from hydra import compose, initialize
from omegaconf.omegaconf import OmegaConf

# from denserwd.models import OctoReward
from denserwd.envs.env_worker import EnvWorkerRM
from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["EMBODIED_PATH"] = os.environ.get(
    "RLINF_DEFAULT_CONFIG_ROOT", os.path.join(REPO_PATH, "examples", "embodiment")
)

SAVE_DATA_DIR = "./data"
CONFIG_DIR = os.environ.get("RLINF_CONFIG_DIR", "./config/vla")


def main(args) -> None:
    initialize(config_path=CONFIG_DIR)
    overrides = []
    if args.resume_dir is not None:
        overrides.append(f"runner.resume_dir={args.resume_dir}")
    if args.log_path is not None:
        overrides.append(f"runner.logger.log_path={args.log_path}")
    cfg = compose(config_name=args.config_name, overrides=overrides)
    cfg = validate_cfg(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "config_name": args.config_name,
                    "resume_dir": cfg_dict["runner"].get("resume_dir"),
                    "log_path": cfg_dict["runner"]["logger"]["log_path"],
                    "experiment_name": cfg_dict["runner"]["logger"]["experiment_name"],
                    "component_placement": cfg_dict["cluster"]["component_placement"],
                },
                indent=2,
            )
        )
        return
    print(json.dumps(cfg_dict, indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # for reward model
    if hasattr(cfg, "reward_model") and cfg.reward_model.use_reward_model:
        use_reward_model = True
    else:
        use_reward_model = False
    if use_reward_model:
        # for debug
        # cfg.reward_model.model.octo_pretrained_path = None

        reward_model_cfg = cfg.reward_model
        reward_model_type = reward_model_cfg.model.model_type
        reward_model_device = reward_model_cfg.device
        reward_model = eval(reward_model_type)(reward_model_cfg).to(reward_model_device)
        reward_model_info = {
            "model": reward_model,
            "model_type": reward_model_type,
            "device": reward_model_device,
            "context_window": reward_model_cfg.model.context_window,
        }
        cfg_dict["env"]["train"]["use_reward_model"] = (
            True  # only for training envs, only save in cfg_dict
        )
        cfg_dict["env"]["train"]["reward_model_info"] = reward_model_info

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")

    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    else:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        actor_worker_cls = EmbodiedFSDPActor
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    if use_reward_model:
        env_group = EnvWorkerRM.create_group(cfg, cfg_dict).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )
    else:
        env_group = EnvWorker.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

    demo_buffer = None
    if cfg.get("data", None):
        from rlinf.data.datasets import create_rl_dataset

        demo_buffer, _ = create_rl_dataset(cfg, tokenizer=None)

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training reward model on LIBERO Benchmark"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        help="Configuration file name",
        default="maniskill_ppo_openvlaoft_quickstart",
    )
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--validate-only", action="store_true")
    # parser.add_argument("--cuda-list", type=str, help="Cuda device list", default="6")
    args = parser.parse_args()
    # args.config_name = "libero_spatial_collect_openpi"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_list
    main(args)
