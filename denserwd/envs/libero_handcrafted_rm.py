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
from typing import Optional, Union

import gym
import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark
from libero.libero.envs import OffScreenRenderEnv
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.libero.utils import (
    get_benchmark_overridden,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    put_info_on_image,
    save_rollout_video,
    tile_images,
    to_tensor,
)
from denserwd.utils.data_utils import batch_image_hwc_to_chw


class LiberoEnvHCRM(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.use_reward_model = cfg.get("use_reward_model", False)
        self.reward_model_info = cfg.get("reward_model_info", None)
        # NOTE: HCRM uses handcrafted reward only.
        # The original reward-model pipeline is kept for compatibility
        # (config fields may still exist), but we intentionally disable it here.
        # if self.use_reward_model:
        #     assert self.reward_model_info is not None
        #     self.reward_model = self.reward_model_info["model"]
        #     self.reward_model_type = self.reward_model_info["model_type"]
        #     self.reward_model_device = self.reward_model_info["device"]
        #     self.context_window = self.reward_model_info["context_window"]
        cfg = OmegaConf.create(cfg)  # transform back to OmegaConf
        self.cfg = cfg

        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []
        self.current_raw_obs = None
        # if self.use_reward_model:
        #     ... allocate reward-model contexts ...
        #     (disabled for handcrafted reward)

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []
        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param):
                seed = param.pop("seed")
                env = OffScreenRenderEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        task_descriptions = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(self.task_descriptions[env_id])
                continue
            task = self.task_suite.get_task(self.task_ids[env_id])
            task_bddl_file = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": task_bddl_file,
                    "seed": self.seed,
                }
            )
            task_descriptions.append(task.language)
        self.task_descriptions = task_descriptions
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

    def update_reset_state_ids(self):
        if self.cfg.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        reset_state_ids = np.arange(self.total_num_group_envs)
        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        self._generator_ordered.shuffle(reset_state_ids)
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (self.num_group,), dtype=int
            )
        else:
            if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
                self.reset_state_ids_all = self.get_reset_state_ids_all()
                self.start_idx = 0
            reset_state_ids = self.reset_state_ids_all[self.seed_offset][
                self.start_idx : self.start_idx + num_reset_states
            ]
            self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        # un-normalized counterpart of self.returns, for diagnostics only
        self.raw_returns = np.zeros(self.num_envs, dtype=np.float32)
        # elapsed_steps at the first success (0 if never succeeded yet)
        self.success_step = np.zeros(self.num_envs, dtype=np.float32)
        self._last_raw_step_reward = np.zeros(self.num_envs, dtype=np.float32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.raw_returns[mask] = 0
            self.success_step[mask] = 0.0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.raw_returns[:] = 0.0
            self.success_step[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        # capture elapsed_steps at the FIRST success only, before success_once
        # is updated below -- episode_len always ends up == the fixed rollout
        # window length (no mid-rollout reset), so it can't be used to measure
        # "how many steps it took to succeed".
        first_success = terminations & (~self.success_once)
        self.success_step = np.where(
            first_success, self.elapsed_steps.astype(np.float32), self.success_step
        )
        self.returns += step_reward
        self.raw_returns += self._last_raw_step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["success_step"] = self.success_step.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["raw_return"] = self.raw_returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        # NOTE: reward is now a sparse, terminal-only signal already normalized
        # by _calc_handcraft_reward (reward = raw_reward / max_steps), so
        # `return` IS the normalized episode reward -- do not divide by
        # episode_len again (that was only meaningful for the old dense
        # per-step -1 scheme). `raw_return` mirrors it in un-normalized units,
        # for diagnostics only -- it is never fed into GRPO advantage calc.
        episode_info["reward"] = episode_info["return"].copy()
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )

        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            if self.task_ids[env_id] != task_ids[j]:
                reconfig_env_idx.append(env_id)
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        init_state = self._get_reset_states(env_idx=env_idx)
        self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(1):  # 15
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}

        # NOTE: reward-model context is intentionally disabled for HCRM.
        # if self.use_reward_model:
        #     self._reset_reward_model_context(env_idx)
        #     self._update_reward_model_obs_context(obs, env_idx)
        #     self._update_reward_model_timestep_context(env_idx)
        #     self._update_reward_model_task_embs()

        return obs, infos

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)

        # NOTE: reward-model predict is intentionally disabled for HCRM.
        # if self.use_reward_model:
        #     self._update_reward_model_act_context(to_tensor(actions), self.env_idx)
        #     data = self._prepare_data_for_reward_model()
        #     data = map_tensor_to_device(data, self.reward_model_device)
        #     with torch.no_grad():
        #         predicted_rewards = self.reward_model.predict(data).cpu().numpy().reshape(-1)

        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        # NOTE: reward-model context update is intentionally disabled for HCRM.
        # if self.use_reward_model:
        #     self._update_reward_model_obs_context(obs, self.env_idx)
        #     self._update_reward_model_timestep_context(self.env_idx)

        step_reward = self._calc_handcraft_reward(terminations, truncations)

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
                "task": self.task_descriptions,
            }
            self.add_new_frames(raw_obs, plot_infos)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )
        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        if self.cfg.is_eval:
            self.update_reset_state_ids()
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    # Fixed failure penalty added on top of max_steps for the raw (un-normalized)
    # episode return. Configurable via cfg.c_fail, defaults to 50.
    DEFAULT_C_FAIL = 50.0

    def _calc_handcraft_reward(self, terminations, truncations):
        """Handcrafted episode reward (numpy float32).

        The whole episode reward is assigned on the terminal step only (all
        non-terminal steps get 0), since our sim only fails by running out of
        max_episode_steps (no early-failure termination):

        raw_reward:
          - success (first occurrence only): -first_success_step
          - failure (truncation without success): -(max_steps + C_fail)

        normalized reward (this is what's returned / fed into training):
          reward = raw_reward / max_steps
          - success: -first_success_step / max_steps, in (-1, 0], closer to 0
            the faster the success (slowest possible success ~= -1.0)
          - failure: -(max_steps + C_fail) / max_steps
            (e.g. max_steps=480, C_fail=50 -> ~= -1.104)

        NOTE: only max_steps is used for normalization (not max_steps+C_fail),
        so the failure reward is intentionally more negative than -1 -- this
        keeps failure strictly worse than even the slowest success.

        NOTE: since training does not auto-reset mid-rollout, `terminations`
        stays latched True on every step after the actual success moment (the
        task condition keeps holding). We must only pay the success reward on
        the step where it first flips False->True (gated on self.success_once,
        which still reflects the *previous* step's state here since
        _record_metrics -- the only place that updates it -- runs after this
        call), otherwise it would be re-applied every remaining step.

        Also stashes the raw (un-normalized) step reward on
        self._last_raw_step_reward for diagnostic logging in _record_metrics;
        it is NOT used for training (only the normalized value returned here
        is, since it's what gets summed into the episode score for GRPO).
        """
        terminations = np.asarray(terminations).astype(bool)
        truncations = np.asarray(truncations).astype(bool)

        max_steps = float(getattr(self.cfg, "max_episode_steps", 1.0))
        c_fail = float(getattr(self.cfg, "c_fail", self.DEFAULT_C_FAIL))
        elapsed_steps = self.elapsed_steps.astype(np.float32)

        raw_step_reward = np.zeros(terminations.shape, dtype=np.float32)

        first_success = terminations & (~self.success_once)
        raw_success_reward = -elapsed_steps
        raw_step_reward = np.where(first_success, raw_success_reward, raw_step_reward)

        raw_fail_reward = -(max_steps + c_fail)

        # NOTE: `terminations` is NOT guaranteed monotonic -- the sim's success
        # condition can flicker True->False->True within an episode (e.g. an
        # object briefly enters then leaves the target region), so a step can
        # read terminations=False at truncation time even though the episode
        # DID succeed earlier. Without the `~self.success_once` guard here,
        # such an episode would get BOTH the success reward (at the flicker
        # step) AND the failure reward (at truncation), silently stacking to
        # something far below the -( max_steps + c_fail)/max_steps floor.
        fail_mask = truncations & (~terminations) & (~self.success_once)
        raw_step_reward = np.where(fail_mask, raw_fail_reward, raw_step_reward)

        raw_step_reward = np.nan_to_num(
            raw_step_reward, nan=-(max_steps + c_fail), posinf=0.0, neginf=-(max_steps + c_fail)
        ).astype(np.float32)

        step_reward = (raw_step_reward / max_steps).astype(np.float32)
        step_reward = np.nan_to_num(
            step_reward, nan=-1.0, posinf=0.0, neginf=-((max_steps + c_fail) / max_steps)
        ).astype(np.float32)

        assert step_reward.dtype == np.float32
        assert np.isfinite(step_reward).all(), (
            f"non-finite reward detected: {step_reward}"
        )

        self._last_raw_step_reward = raw_step_reward
        return step_reward

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def add_new_frames(self, raw_obs, plot_infos):
        images = []
        for env_id, raw_single_obs in enumerate(raw_obs):
            info_item = {
                k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()
            }
            img = raw_single_obs["agentview_image"][::-1, ::-1]
            img = put_info_on_image(img, info_item)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []

    def _update_reward_model_obs_context(self, obs, env_idx):
        if not self.use_reward_model:
            return
        main_images = batch_image_hwc_to_chw(obs["main_images"]) / 255.0
        wrist_images = batch_image_hwc_to_chw(obs["wrist_images"]) / 255.0
        robot_states = obs["states"]
        for env_id in env_idx:
            self.main_images_context[env_id] = torch.roll(
                self.main_images_context[env_id], shifts=-1, dims=0
            )
            self.main_images_context[env_id][-1] = main_images[env_id]
            self.wrist_images_context[env_id] = torch.roll(
                self.wrist_images_context[env_id], shifts=-1, dims=0
            )
            self.wrist_images_context[env_id][-1] = wrist_images[env_id]
            self.robot_states_context[env_id] = torch.roll(
                self.robot_states_context[env_id], shifts=-1, dims=0
            )
            self.robot_states_context[env_id][-1] = robot_states[env_id]

    def _update_reward_model_act_context(self, actions, env_idx):
        if not self.use_reward_model:
            return
        for env_id in env_idx:
            self.actions_context[env_id] = torch.roll(
                self.actions_context[env_id], shifts=-1, dims=0
            )
            self.actions_context[env_id][-1] = actions[env_id]

    def _update_reward_model_timestep_context(self, env_idx):
        if not self.use_reward_model:
            return
        for env_id in env_idx:
            last_timestep = self.timesteps_context[env_id][-1]
            self.timesteps_context[env_id] = torch.roll(
                self.timesteps_context[env_id], shifts=-1, dims=0
            )
            self.timesteps_context[env_id][-1] = last_timestep + 1
    
    def _update_reward_model_task_embs(self):
        if not self.use_reward_model:
            return
        # task embeds need to be updated together
        self.task_embs = self.reward_model.get_task_embs(self.task_descriptions)

    def _reset_reward_model_context(self, env_idx):
        if not self.use_reward_model:
            return
        for env_id in env_idx:
            torch.zero_(self.main_images_context[env_id])
            torch.zero_(self.wrist_images_context[env_id])
            torch.zero_(self.robot_states_context[env_id])
            torch.zero_(self.actions_context[env_id])
            torch.zero_(self.timesteps_context[env_id])
    
    def _prepare_data_for_reward_model(self):
        task_embeds = torch.stack(self.task_embs, dim=0)
        main_images = torch.stack(self.main_images_context, dim=0)
        wrist_images = torch.stack(self.wrist_images_context, dim=0)
        robot_states = torch.stack(self.robot_states_context, dim=0)
        actions = torch.stack(self.actions_context, dim=0)
        timesteps = torch.stack(self.timesteps_context, dim=0)
        data = {
            "obs": {
                "main_images": main_images,
                "wrist_images": wrist_images,
                "robot_states": robot_states,
            },
            "task_emb": task_embeds,
            "actions": actions,
            "timesteps": timesteps,
        }
        return data
