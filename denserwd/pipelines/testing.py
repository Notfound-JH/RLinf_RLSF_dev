"""
TestPipeline for evaluating trained reward models.

Features:
1. Load trained model from checkpoint
2. Evaluate on boundary dataset with metrics
3. Predict rewards for demos and generate visualization videos
"""

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from denserwd.utils.tensor_utils import map_tensor_to_device
from denserwd.utils.data_utils import wrap_lerobot_batch_for_reward_model


class TestingPipeline:
    """
    Pipeline for testing trained reward models.

    Supports:
    - Loading models from checkpoints
    - Evaluating on datasets
    - Visualizing predictions with videos
    """

    def __init__(
        self,
        model,
        cfg,
        device="cuda",
    ):
        """
        Initialize the TestPipeline.

        Args:
            model: The reward model
            cfg: Configuration object
            device: Device to use for testing
        """
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.model.eval()

        # Test parameters
        self.batch_size = cfg.test.get("batch_size", 32) if cfg.test else 32
        self.num_workers = cfg.test.get("num_workers", 4) if cfg.test else 4

        print(f"Initialized on device: {device}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if checkpoint_path is None or checkpoint_path == "":
            print(f"No checkpoint path provided, skipp loading model.")
            return
        print(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)

        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.modules_to_save.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model state dict")
        else:
            print(f"Warning: No model_state_dict found in checkpoint")

        # Load additional info
        if 'current_step' in checkpoint:
            print(f"Checkpoint step: {checkpoint['current_step']}")
        if 'best_loss' in checkpoint:
            print(f"Checkpoint best loss: {checkpoint['best_loss']:.4f}")

        print(f"Checkpoint loaded successfully!")
    
    def test_voc_on_lerobot_dataset(self, lerobot_dataloader) -> Dict: 
        dataloader = lerobot_dataloader
        print(f"Dataset size: {len(dataloader)} batches")

        task_frame_index, task_pred_reward = {}, {}  # organized by {task_index: {episode_index: frame_index_list/pred_reward_list}}}
        task_frame_index_reverse, task_pred_reward_reverse = {}, {}  # organized by {task_index: {episode_index: frame_index_list/pred_reward_list}}}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                batch = map_tensor_to_device(batch, device=self.device)

                if "reversed_samples" in batch:
                    batch_reverse = batch.pop("reversed_samples")
                else:
                    batch_reverse = None

                # predict rewards
                batch = wrap_lerobot_batch_for_reward_model(batch)
                pred_rewards = self.model.predict(batch).squeeze(-1)  # (B,)

                # store frame indices and predicted rewards by task index
                for i in range(len(batch['task_index'])):
                    task_idx = batch['task_index'][i].item()
                    ep_index = batch['episode_index'][i].item()
                    frame_idx = batch['frame_index'][i].item()
                    pred_reward = pred_rewards[i].item()
                    if task_idx not in task_frame_index:
                        task_frame_index[task_idx] = {}
                        task_pred_reward[task_idx] = {}
                    if ep_index not in task_frame_index[task_idx]:
                        task_frame_index[task_idx][ep_index] = []
                        task_pred_reward[task_idx][ep_index] = []
                    task_frame_index[task_idx][ep_index].append(frame_idx)
                    task_pred_reward[task_idx][ep_index].append(pred_reward)
                
                if batch_reverse is not None:
                    # predict rewards
                    batch_reverse = wrap_lerobot_batch_for_reward_model(batch_reverse)
                    pred_rewards = self.model.predict(batch_reverse).squeeze(-1)  # (B,)

                    # store frame indices and predicted rewards by task index
                    for i in range(len(batch_reverse['task_index'])):
                        task_idx = batch_reverse['task_index'][i].item()
                        ep_index = batch_reverse['episode_index'][i].item()
                        frame_idx = batch_reverse['frame_index'][i].item()
                        pred_reward = pred_rewards[i].item()
                        if task_idx not in task_frame_index_reverse:
                            task_frame_index_reverse[task_idx] = {}
                            task_pred_reward_reverse[task_idx] = {}
                        if ep_index not in task_frame_index_reverse[task_idx]:
                            task_frame_index_reverse[task_idx][ep_index] = []
                            task_pred_reward_reverse[task_idx][ep_index] = []
                        task_frame_index_reverse[task_idx][ep_index].append(frame_idx)
                        task_pred_reward_reverse[task_idx][ep_index].append(pred_reward)

        all_voc, all_vroc = [], []
        task_voc, task_vroc = {}, {}
        for task_idx in task_frame_index.keys():
            if task_idx not in task_voc:
                task_voc[task_idx] = {}
                task_vroc[task_idx] = {}
            for ep_index in task_frame_index[task_idx].keys():
                frame_indices = np.array(task_frame_index[task_idx][ep_index])
                pred_rewards = np.array(task_pred_reward[task_idx][ep_index])
                frame_indices = np.arange(len(frame_indices))  # use relative frame index for evaluation
                pred_rewards_indices = np.argsort(pred_rewards)
                # compute voc and vroc
                voc, _ = spearmanr(pred_rewards_indices, frame_indices)
                task_voc[task_idx][ep_index] = voc
                all_voc.append(voc)

                if task_idx not in task_frame_index_reverse or ep_index not in task_frame_index_reverse[task_idx] \
                    or len(task_frame_index_reverse[task_idx][ep_index]) == 0:
                    # if no reversed samples for this task, use forward samples to compute vroc as well
                    vroc, _ = spearmanr(pred_rewards_indices[::-1], frame_indices[::-1])
                    task_vroc[task_idx][ep_index] = vroc
                    all_vroc.append(vroc)

        if len(task_frame_index_reverse) > 0:
            for task_idx in task_frame_index_reverse.keys():
                if task_idx not in task_vroc:
                    task_vroc[task_idx] = {}
                for ep_index in task_frame_index_reverse[task_idx].keys():
                    frame_indices = np.array(task_frame_index_reverse[task_idx][ep_index])
                    pred_rewards = np.array(task_pred_reward_reverse[task_idx][ep_index])
                    frame_indices = np.arange(len(frame_indices))[::-1]  # use relative frame index for evaluation
                    pred_rewards_indices = np.argsort(pred_rewards)
                    # compute vroc
                    vroc, _ = spearmanr(pred_rewards_indices, frame_indices)
                    task_vroc[task_idx][ep_index] = vroc
                    all_vroc.append(vroc)

        mean_voc, mean_vroc = np.mean(all_voc), np.mean(all_vroc)
        std_voc, std_vroc = np.std(all_voc), np.std(all_vroc)
        metrics = {
            'mean_voc': mean_voc,
            'std_voc': std_voc,
            'mean_vroc': mean_vroc,
            'std_vroc': std_vroc,
        }

        # Print metrics
        print(f"\nEvaluation Results:")
        print(f"  Mean VOC: {metrics['mean_voc']:.4f}")
        print(f"  Std VOC: {metrics['std_voc']:.4f}")
        print(f"  Mean VROC: {metrics['mean_vroc']:.4f}")
        print(f"  Std VROC: {metrics['std_vroc']:.4f}")

        return metrics, task_voc, task_vroc, {'frame_indices': task_frame_index, 
                                              'pred_rewards': task_pred_reward,
                                              'frame_indices_reverse': task_frame_index_reverse,
                                              'pred_rewards_reverse': task_pred_reward_reverse}

    def test_on_boundary_dataset(self, boundary_dataloader) -> Dict:
        """
        Test model on a boundary dataset and compute metrics.

        Args:
            boundary_dataset: Dataloader object

        Returns:
            Dictionary with evaluation metrics
        """
        dataloader = boundary_dataloader
        print(f"Dataset size: {len(dataloader)} batches")

        # Evaluation loop
        total_mse_losses = []
        total_rmse_losses = []
        total_mae_losses = []
        total_preds = []
        total_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                batch = map_tensor_to_device(batch, device=self.device)

                # Forward pass
                pred_rewards = self.model.predict(batch).squeeze(-1)  # (B,)
                target_rewards = batch['rewards'][:, -1]  # (B,)

                # Compute metrics
                mse_loss = F.mse_loss(pred_rewards, target_rewards)
                rmse_loss = torch.sqrt(mse_loss)
                mae_loss = torch.abs(pred_rewards - target_rewards).mean()

                total_mse_losses.append(mse_loss.item())
                total_rmse_losses.append(rmse_loss.item())
                total_mae_losses.append(mae_loss.item())
                total_preds.append(pred_rewards.cpu().numpy())
                total_targets.append(target_rewards.cpu().numpy())

        # Compute aggregate metrics
        all_preds = np.concatenate(total_preds)
        all_targets = np.concatenate(total_targets)

        metrics = {
            'mse': float(np.mean(total_mse_losses)),
            'rmse': float(np.mean(total_rmse_losses)),
            'mae': float(np.mean(total_mae_losses)),
            'pred_mean': float(all_preds.mean()),
            'pred_std': float(all_preds.std()),
            'pred_min': float(all_preds.min()),
            'pred_max': float(all_preds.max()),
            'target_mean': float(all_targets.mean()),
            'target_std': float(all_targets.std()),
            'num_samples': len(all_preds),
        }

        # Print metrics
        print(f"\nEvaluation Results:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Predictions: {metrics['pred_mean']:.3f} ± {metrics['pred_std']:.3f} "
              f"[{metrics['pred_min']:.3f}, {metrics['pred_max']:.3f}]")
        print(f"  Targets: {metrics['target_mean']:.3f} ± {metrics['target_std']:.3f}")

        return metrics

    def predict_demo_rewards(self, dataset) -> np.ndarray:
        """
        Predict rewards for all timesteps in a dataset.

        Args:
            dataset: Dataset object
        Returns:
            Array of predicted rewards with shape (T,)
        """
        print(f"Predicting rewards for demo...")
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # remember to set to false
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )
        rewards = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting rewards"):
                # Move to device
                batch = map_tensor_to_device(batch, device=self.device)

                # Forward pass
                pred_rewards = self.model.predict(batch).squeeze(-1)  # (B,)
                rewards.extend(pred_rewards.cpu().tolist())
        rewards = np.array(rewards)
        return rewards
