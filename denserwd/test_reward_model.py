import os
import argparse
import pprint
import wandb
import yaml
import torch

from easydict import EasyDict
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from denserwd.models import OctoReward
from denserwd.datasets.datasets import get_dataset, SequenceVLDataset
from denserwd.pipelines import TestingPipeline
from denserwd.utils.utils import get_hdf5_files, control_seed
from denserwd.utils.data_utils import load_demo_data
from denserwd.scripts.visualize_demo import create_visualization_video


os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONFIG_DIR = "./config"
RESULT_DIR = "./result"


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

    ### for debug
    cfg.model.octo_pretrained_path = None
    args.data_dir = "/mnt/storage/users/dwu_data/vla/libero_spatial_collect_openpi/"

    data_files = get_hdf5_files(args.data_dir)
    # n_manip_tasks = len(data_files)
    n_manip_tasks = 1

    # prepare datasets from the benchmark
    tasks = []

    trajs_per_task = 50  # currently we assume all tasks have 50 trajs
    train_traj_num = int(trajs_per_task * cfg.data.train_dataset_ratio)
    eval_traj_num = int(trajs_per_task * cfg.data.eval_dataset_ratio)
    test_traj_keys = [f'traj_{i}' for i in range(train_traj_num + eval_traj_num, trajs_per_task)]

    # 1. Test on boundary datsets for all tasks and trajs
    print("=" * 60)
    print("Stage 1: Testing on Boundary Datasets")
    print("=" * 60)
    test_boundary_datasets = []
    for i in range(n_manip_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_boundary_dataset, task = get_dataset(
                dataset_path=data_files[i],
                obs_modality=cfg.data.obs.modality,
                seq_len=cfg.model.context_window,
                traj_keys=test_traj_keys,
                add_reward=True,
                boundary_k=cfg.data.boundary_k,
                sample_step=1
            )
            test_boundary_datasets.append(task_i_boundary_dataset)
            tasks.append(task)
        except Exception as e:
            print(
                f"[error] failed to load dataset for task {i} from {data_files[i]}"
            )
            print(f"[error] {e}")
        
    # create model
    model = eval(cfg.model.model_type)(cfg)

    # create dateset
    task_embs = model.get_task_embs(tasks)

    test_boundary_datasets = ConcatDataset([SequenceVLDataset(ds, emb) for (ds, emb) in zip(test_boundary_datasets, task_embs)])

    if cfg.use_wandb:
        wandb.init(project="libero", config=cfg)
        wandb.run.name = cfg.experiment_name

    pipeline = eval(cfg.test.pipeline)(
        model=model,
        cfg=cfg,
        device=cfg.device,
    )
    dataloader = DataLoader(
        test_boundary_datasets,
        batch_size=cfg.test.batch_size,
        shuffle=False,  # no shuffling for evaluation
        num_workers=cfg.test.num_workers,
        persistent_workers=True if cfg.test.num_workers > 0 else False,
    )
    pipeline.load_checkpoint(cfg.test.checkpoint_path)
    pipeline.test_on_boundary_dataset(test_boundary_datasets)

    # 2. Test on all timesteps and visualize
    print("=" * 60)
    print("Stage 2: Testing on All Timesteps and Visualization")
    print("=" * 60)
    output_dir = os.path.join(RESULT_DIR, "videos", args.config_name, args.label)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lask_k = 1
    test_traj_keys = [f'traj_{i}' for i in range(trajs_per_task - lask_k, trajs_per_task)]  # only visualize the last lask_k traj for each task
    for i in range(n_manip_tasks):
        traj_path = data_files[i]
        for traj_key in test_traj_keys:  # one traj at a time
            try:
                task_i_dataset, task = get_dataset(
                    dataset_path=traj_path,
                    obs_modality=cfg.data.obs.modality,
                    seq_len=cfg.model.context_window,
                    traj_keys=[traj_key],
                    add_reward=True,
                    boundary_k=5,
                    sample_step=1
                )
                task_i_dataset = SequenceVLDataset(task_i_dataset, task_embs[i])
                predicted_rewards = pipeline.predict_demo_rewards(task_i_dataset)
                traj_data = load_demo_data(traj_path, traj_key, image_key="main_images", reward_key="rewards")
                output_video_path = os.path.join(output_dir, f"{tasks[i].replace(' ', '_')}_{traj_key}_visualization.mp4")
                create_visualization_video(
                    demo_data=traj_data, 
                    output_path=output_video_path, 
                    predicted_rewards=predicted_rewards
                )
            except Exception as e:
                print(
                    f"[error] failed to visualize task {i} name '{tasks[i]}' traj {traj_key}"
                )
                print(f"[error] {e}")
                raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training reward model on LIBERO Benchmark")
    parser.add_argument("--config_name", type=str, help="Configuration file name", default="octo_default")  
    parser.add_argument("--label", type=str, help="Label to distinguish different test experiments", default="")  
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint") 
    parser.add_argument("--data_dir", type=str, help="Directory containing the dataset") 
    parser.add_argument("--device", type=str, default='cuda:6' if torch.cuda.is_available() else 'cpu') 

    # arguments for other settings
    parser.add_argument("--seed", type=int, help="Random seed", default=42)

    args = parser.parse_args()
    # args.config_name = "octo_base_tu500_tf"
    # args.device = 'cpu'
    test(args)
