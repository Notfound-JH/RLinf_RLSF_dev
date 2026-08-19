import os
import argparse
import pprint
import wandb
import yaml
import torch
import numpy as np

from easydict import EasyDict
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

from denserwd.models import OctoReward, Qwen3VLReward
from denserwd.datasets.datasets import get_dataset, SequenceVLDataset
from denserwd.pipelines import TrainingPipeline
from denserwd.utils.utils import get_hdf5_files, control_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONFIG_DIR = "./config/reward_model"
RESULT_DIR = "data/rwd_models"
TASK_SUITE_TO_DATA_DIR = {
    'libero_spatial': "data/datasets/libero_spatial_collect_openpi",
    'libero_object': "data/datasets/libero_object_collect_openpi",
    'libero_goal': "data/datasets/libero_goal_collect_openpi",
    'libero_10': "data/datasets/libero_10_collect_openpi",
}


def train(args):
    # preprocessing
    initialize(config_path=CONFIG_DIR)
    hydra_cfg = compose(config_name=args.config_name)
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    cfg.device = args.device
    cfg.seed = args.seed

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # control seed
    control_seed(cfg.seed)

    ## for debug
    # cfg.train.n_epochs = 1
    # cfg.train.curriculum_epochs = 1
    # cfg.train.eval_every = 1
    # cfg.train.batch_size = 2
    # cfg.train.train_target_first = False
    # cfg.model.octo_pretrained_path = None
    # args.device = "cuda:2"
    # args.data_dir = "data/datasets/libero_spatial_collect_openpi2/"

    if args.task_suite in TASK_SUITE_TO_DATA_DIR:
        data_files = get_hdf5_files(TASK_SUITE_TO_DATA_DIR[args.task_suite])
    elif args.task_suite == 'libero_40':
        data_files = []
        for suite in TASK_SUITE_TO_DATA_DIR.keys():
            if suite.startswith('libero'):
                data_files += get_hdf5_files(TASK_SUITE_TO_DATA_DIR[suite])
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite}")
    n_manip_tasks = len(data_files)
    # n_manip_tasks = 2

    # prepare datasets 
    train_datasets = []
    train_boundary_datasets = []
    eval_datasets = []
    tasks = []
    shape_meta = None

    trajs_per_task = 50  # currently we assume all tasks have 50 trajectories
    traj_keys = [f'traj_{i}' for i in range(trajs_per_task)]
    
    # train_traj_num = int(trajs_per_task * cfg.data.train_dataset_ratio)
    # eval_traj_num = trajs_per_task - train_traj_num
    # train_traj_keys = traj_keys[:train_traj_num]
    # eval_traj_keys = traj_keys[train_traj_num:]

    # use a different strategy: training on all, eval on portion
    np.random.shuffle(traj_keys)
    train_traj_keys = traj_keys
    eval_traj_keys = traj_keys[:int(trajs_per_task * cfg.data.eval_dataset_ratio)]

    for i in range(n_manip_tasks):
        try:
            task_i_dataset, task = get_dataset(
                dataset_path=data_files[i],
                obs_modality=cfg.data.obs.modality,
                seq_len=cfg.model.context_window,
                traj_keys=train_traj_keys,
                add_reward=True,
                boundary_k=cfg.data.boundary_k,
                sample_step=cfg.data.sample_step
            )
            task_i_boundary_dataset, _ = get_dataset(
                dataset_path=data_files[i],
                obs_modality=cfg.data.obs.modality,
                seq_len=cfg.model.context_window,
                traj_keys=train_traj_keys,  
                add_reward=True,
                boundary_k=1,
                sample_step=1,
                use_boundary=True,
            )
            task_i_eval_dataset, _ = get_dataset(
                dataset_path=data_files[i],
                obs_modality=cfg.data.obs.modality,
                seq_len=cfg.model.context_window,
                traj_keys=eval_traj_keys,
                add_reward=True,
                boundary_k=1,  # by default, we use boundary dataset for evaluation
                sample_step=1,
                use_boundary=True,
            )
            train_datasets.append(task_i_dataset)
            train_boundary_datasets.append(task_i_boundary_dataset)
            eval_datasets.append(task_i_eval_dataset)
            tasks.append(task)
        except Exception as e:
            print(
                f"[error] failed to load dataset for task {i} from {data_files[i]}"
            )
            print(f"[error] {e}")
        
    # create model
    model = eval(cfg.model.model_type)(cfg)

    # create dataset side task conditioning
    task_embs = model.get_task_embs(tasks)
    if hasattr(model, 'set_task_prompts'):
        model.set_task_prompts(tasks)

    train_datasets = ConcatDataset([SequenceVLDataset(ds, emb) for (ds, emb) in zip(train_datasets, task_embs)])
    train_boundary_datasets = ConcatDataset([SequenceVLDataset(ds, emb) for (ds, emb) in zip(train_boundary_datasets, task_embs)])
    eval_datasets = ConcatDataset([SequenceVLDataset(ds, emb) for (ds, emb) in zip(eval_datasets, task_embs)])

    if cfg.use_wandb:
        wandb.init(project="libero", config=cfg)
        wandb.run.name = cfg.experiment_name

    pipeline = eval(cfg.train.pipeline)(
        model=model,
        cfg=cfg,
        dataset=train_datasets,
        boundary_dataset=train_boundary_datasets,
        eval_dataset=eval_datasets,
        device=cfg.device,
    )
    print("Start training reward model...")
    if cfg.train.resume:
        print("Resume training from:", cfg.train.resume_path)
        pipeline.load(cfg.train.resume_path)
    pipeline.train(eval_every=cfg.train.eval_every, save_every=cfg.train.save_every,
                   save_dir=os.path.join(RESULT_DIR, "models", args.config_name), start_epoch=cfg.train.start_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training reward model on LIBERO Benchmark")
    parser.add_argument("--config-name", type=str, help="Configuration file name", default="default")  
    parser.add_argument("--device", type=str, default='cuda:6' if torch.cuda.is_available() else 'cpu') 
    parser.add_argument("--task-suite", type=str, help="Task suite", default="libero_40") 

    # arguments for other settings
    parser.add_argument("--seed", type=int, help="Random seed", default=42)

    args = parser.parse_args()
    # args.config_name = "octo_default"
    # args.device = 'cuda:2'
    train(args)
