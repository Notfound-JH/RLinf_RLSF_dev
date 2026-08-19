from typing import Any
from omegaconf import DictConfig
from easydict import EasyDict
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.envs import get_env_cls
from rlinf.envs.wrappers import RecordVideo
from denserwd.envs.libero_rm import LiberoEnvRM
from denserwd.envs.libero_handcrafted_rm import LiberoEnvHCRM

class EnvWorkerRM(EnvWorker):
    def __init__(self, cfg: DictConfig, cfg_rm: EasyDict):
        super().__init__(cfg)
        self.cfg_rm = cfg_rm

    def init_worker(self):
        self.enable_offload = self.cfg.env.enable_offload

        train_env_cls = get_env_cls(
            self.cfg.env.train.env_type, self.cfg.env.train
        )
        eval_env_cls = get_env_cls(
            self.cfg.env.eval.env_type, self.cfg.env.eval
        )

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        if not self.only_eval:
            for stage_id in range(self.stage_num):
                if train_env_cls is LiberoEnvRM:
                    # Pass reward model config to LiberoEnvRM
                    env = train_env_cls(
                        cfg=self.cfg_rm["env"]["train"],
                        num_envs=self.train_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        worker_info=self.worker_info,
                    )
                else:
                    env = train_env_cls(
                        cfg=self.cfg.env.train,
                        num_envs=self.train_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        worker_info=self.worker_info,
                    )
                if self.cfg.env.train.video_cfg.save_video:
                    env = RecordVideo(env, self.cfg.env.train.video_cfg)
                self.env_list.append(env)
                
        if self.enable_eval:
            for stage_id in range(self.stage_num):
                if eval_env_cls is LiberoEnvHCRM:
                    # Pass handcrafted reward config to LiberoEnvHCRM
                    env = eval_env_cls(
                        cfg=self.cfg_rm["env"]["eval"],
                        num_envs=self.eval_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        worker_info=self.worker_info,
                    )
                else:
                    env = eval_env_cls(
                        cfg=self.cfg.env.eval,
                        num_envs=self.eval_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        worker_info=self.worker_info,
                    )
                if self.cfg.env.eval.video_cfg.save_video:
                    env = RecordVideo(env, self.cfg.env.eval.video_cfg)
                self.eval_env_list.append(env)

        if not self.only_eval:
            self._init_env()



# class EnvWorkerHCRM(EnvWorker):
#     def __init__(self, cfg: DictConfig, cfg_rm: EasyDict):
#         super().__init__(cfg)
#         self.cfg_rm = cfg_rm

#     def init_worker(self):
#         self.enable_offload = self.cfg.env.enable_offload

#         train_env_cls = get_env_cls(
#             self.cfg.env.train.env_type, self.cfg.env.train
#         )
#         eval_env_cls = get_env_cls(
#             self.cfg.env.eval.env_type, self.cfg.env.eval
#         )

#         # This is a barrier to ensure all envs' initial setup upon import is done
#         # Essential for RealWorld env to ensure initial ROS node setup is done
#         self.broadcast(
#             True,
#             groups=[(self._group_name, list(range(self._world_size)))],
#         )

#         if not self.only_eval:
#             for stage_id in range(self.stage_num):
#                 if train_env_cls is LiberoEnvHCRM:
#                     # Pass handcrafted reward config to LiberoEnvHCRM
#                     env = train_env_cls(
#                         cfg=self.cfg_rm["env"]["train"],
#                         num_envs=self.train_num_envs_per_stage,
#                         seed_offset=self._rank * self.stage_num + stage_id,
#                         total_num_processes=self._world_size * self.stage_num,
#                         worker_info=self.worker_info,
#                     )
#                 else:
#                     env = train_env_cls(
#                         cfg=self.cfg.env.train,
#                         num_envs=self.train_num_envs_per_stage,
#                         seed_offset=self._rank * self.stage_num + stage_id,
#                         total_num_processes=self._world_size * self.stage_num,
#                         worker_info=self.worker_info,
#                     )
#                 if self.cfg.env.train.video_cfg.save_video:
#                     env = RecordVideo(env, self.cfg.env.train.video_cfg)
#                 self.env_list.append(env)
                
#         if self.enable_eval:
#             for stage_id in range(self.stage_num):
#                 if eval_env_cls is LiberoEnvHCRM:
#                     # Pass handcrafted reward config to LiberoEnvHCRM
#                     env = eval_env_cls(
#                         cfg=self.cfg_rm["env"]["eval"],
#                         num_envs=self.eval_num_envs_per_stage,
#                         seed_offset=self._rank * self.stage_num + stage_id,
#                         total_num_processes=self._world_size * self.stage_num,
#                         worker_info=self.worker_info,
#                     )
#                 else:
#                     env = eval_env_cls(
#                         cfg=self.cfg.env.eval,
#                         num_envs=self.eval_num_envs_per_stage,
#                         seed_offset=self._rank * self.stage_num + stage_id,
#                         total_num_processes=self._world_size * self.stage_num,
#                         worker_info=self.worker_info,
#                     )
#                 if self.cfg.env.eval.video_cfg.save_video:
#                     env = RecordVideo(env, self.cfg.env.eval.video_cfg)
#                 self.eval_env_list.append(env)

#         if not self.only_eval:
#             self._init_env()
