import os
import argparse
import pprint
import json
import yaml
import torch
import numpy as np

from easydict import EasyDict
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from denserwd.models import OctoReward
from denserwd.datasets.lerobot_dataset import LeRobotDatasetRM
from denserwd.pipelines import TestingPipeline
from denserwd.utils.utils import control_seed
from denserwd.utils.dataset_utils import create_torch_dataset
from denserwd.scripts.visualize_demo import create_visualization_video
from denserwd.datasets.libero_config import TASK_IDX_TO_DESCRIPTIONS, TASK_IDX_TO_SUITE


os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONFIG_DIR = "./config/reward_model"
RESULT_DIR = "./data/results"


def test(args):
    # preprocessing
    initialize(config_path=CONFIG_DIR)
    hydra_cfg = compose(config_name=args.config_name)
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    cfg.device = args.device
    cfg.seed = args.seed
    if args.checkpoint_path is not None:
        cfg.test.checkpoint_path = args.checkpoint_path

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # control seed
    control_seed(cfg.seed)

    # for debug
    # cfg.model.octo_pretrained_path = None

    # create model
    model = eval(cfg.model.model_type)(cfg)

    # create dateset
    config = get_openpi_config(
        cfg.test.openpi.config_name,
        model_path=cfg.test.openpi.model_path,
        batch_size=cfg.test.batch_size,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    episode_index_path = cfg.test.data.get("episode_index_path", None)
    if episode_index_path is not None:
        episodes = np.loadtxt(episode_index_path, dtype=np.int32, delimiter=',').tolist()
    else:
        episodes = None
    
    # for debug
    # episodes = [0, 1]
    # cfg.test.batch_size = 32

    context_window = cfg.model.context_window
    add_reverse = cfg.test.get("add_reverse", True)
    sample_step = cfg.test.get("sample_step", 1)
    dataset = create_torch_dataset(data_config, model_config=config.model, action_horizon=config.model.action_horizon, 
                                   dataset_cls=LeRobotDatasetRM, episodes=episodes, context_window=context_window,
                                   add_reverse=add_reverse, sample_step=sample_step)
    libero_tasks = list(TASK_IDX_TO_DESCRIPTIONS.values())
    task_embs = model.get_task_embs(libero_tasks)
    dataset.set_task_embs(task_embs)

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        persistent_workers=True if cfg.test.num_workers > 0 else False,
    )

    # create pipeline
    pipeline = eval(cfg.test.pipeline)(
        model=model,
        cfg=cfg,
        device=cfg.device,
    )
    pipeline.load_checkpoint(cfg.test.checkpoint_path)

    # start testing
    overall_metrics, task_voc, task_vroc, info = pipeline.test_voc_on_lerobot_dataset(dataloader)

    # organize results by task suites
    suite_voc, suite_vroc = {}, {}
    suite_all_voc, suite_all_vroc = {}, {}
    for task_idx in task_voc.keys():
        task_suite = TASK_IDX_TO_SUITE[task_idx]
        if task_suite not in suite_voc:
            suite_voc[task_suite] = []
            suite_vroc[task_suite] = []
            suite_all_voc[task_suite] = []
            suite_all_vroc[task_suite] = []
        suite_voc[task_suite].append(task_voc[task_idx])
        suite_vroc[task_suite].append(task_vroc[task_idx])
        for ep_index in task_voc[task_idx].keys():
            suite_all_voc[task_suite].append(task_voc[task_idx][ep_index])
            suite_all_vroc[task_suite].append(task_vroc[task_idx][ep_index])
    
    suite_metrics = {}
    for suite in suite_voc.keys():
        suite_metrics[suite] = {
            'mean_voc': np.mean(suite_all_voc[suite]),
            'std_voc': np.std(suite_all_voc[suite]),
            'mean_vroc': np.mean(suite_all_vroc[suite]),
            'std_vroc': np.std(suite_all_vroc[suite]),
        }
    print("Suite Metrics:")
    for suite, metrics in suite_metrics.items():
        print(f"  Suite: {suite}")
        print(f"    Mean VOC: {metrics['mean_voc']:.4f}")
        print(f"    Std VOC: {metrics['std_voc']:.4f}")
        print(f"    Mean VROC: {metrics['mean_vroc']:.4f}")
        print(f"    Std VROC: {metrics['std_vroc']:.4f}")
        
    output_dir = os.path.join(RESULT_DIR, args.config_name + f'_ss{sample_step}', args.label)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"voc_vroc_results_{cfg.test.checkpoint_path.split('/')[-1][:-4]}.json")

    json.dump({"overall_metrics": overall_metrics, 
            "suite_metrics": suite_metrics,
            "task_voc": task_voc,
            "task_vroc": task_vroc,
            "suite_voc": suite_voc,
            "suite_vroc": suite_vroc,
            **info}, 
            open(output_path, "w"), indent=2)
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training reward model on LIBERO Benchmark")
    parser.add_argument("--config-name", type=str, help="Configuration file name", default="libero_40_lerobot_eval")  
    parser.add_argument("--label", type=str, help="Label to distinguish different test experiments", default="")  
    parser.add_argument("--checkpoint-path", type=str, help="Path to model checkpoint") 
    parser.add_argument("--device", type=str, default='cuda:7' if torch.cuda.is_available() else 'cpu') 

    # arguments for other settings
    parser.add_argument("--seed", type=int, help="Random seed", default=42)

    args = parser.parse_args()
    # args.config_name = "octo_base_tu500_tf"
    # args.device = 'cpu'
    # args.checkpoint_path = "/workspace/RLinf/denserwd/data/rwd_models/models/libero_40/checkpoint_epoch_0.pth"
    test(args)
