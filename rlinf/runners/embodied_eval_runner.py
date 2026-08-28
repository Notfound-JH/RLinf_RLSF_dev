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

import json
import os
import typing

import numpy as np

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        report_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
        )
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "eval_report.json")
        serializable_metrics = {
            key: value.item()
            if isinstance(value, (np.generic, np.ndarray))
            else value
            for key, value in eval_metrics.items()
        }
        report = {
            "overall_success_rate": serializable_metrics.get("eval/success_rate"),
            "num_trajectories": serializable_metrics.get("eval/num_trajectories"),
            "per_task": {},
            "metrics": serializable_metrics,
        }
        task_suite = None
        try:
            from rlinf.envs.libero.utils import get_benchmark_overridden

            suite_name = self.cfg.env.eval.get("task_suite_name")
            if suite_name:
                task_suite = get_benchmark_overridden(suite_name)()
        except (ImportError, KeyError, TypeError, ValueError):
            self.logger.warning("Could not load LIBERO task names for the report.")

        task_prefix = "eval/task_"
        task_suffix = "_success_rate"
        for key, success_rate in serializable_metrics.items():
            if not (key.startswith(task_prefix) and key.endswith(task_suffix)):
                continue
            task_id = int(key[len(task_prefix) : -len(task_suffix)])
            task_name = f"task_{task_id}"
            if task_suite is not None:
                task_name = task_suite.get_task(task_id).language
            report["per_task"][str(task_id)] = {
                "task_name": task_name,
                "success_rate": success_rate,
                "num_trajectories": serializable_metrics.get(
                    f"eval/task_{task_id}_num_trajectories"
                ),
            }
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, sort_keys=True)
            report_file.write("\n")
        self.logger.info(f"Saved evaluation report to {report_path}.")

        self.metric_logger.finish()
