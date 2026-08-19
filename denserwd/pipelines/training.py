"""
TrainingPipeline for training reward model.

Implements TD(0)-inspired training with target network synchronization.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from denserwd.utils.tensor_utils import map_tensor_to_device


class TrainingPipeline:
    """
    Pipeline for training reward model with curriculum learning.

    Features:
    1. TD(0)-style target network for computing next-step rewards
    2. Periodic synchronization between main and target networks
    3. Episode-aware loss computation (r_T for terminal, target prediction otherwise)
    4. Curriculum training (start with t=0 and t=T only)
    5. Entropy regularization for stable training
    """

    def __init__(
        self,
        model,
        cfg,
        dataset,
        boundary_dataset=None,
        eval_dataset=None,
        device="cuda",
    ):
        """
        Initialize the CurriculumTrainingPipeline.

        Args:
            model: The reward model (f network)
            dataset: Dataset containing demonstrations
            boundary_dataset: Dataset for curriculum training (optional)
            eval_dataset: Dataset for evaluation (optional)
            cfg: Configuration object with attributes:
                - train.batch_size: Batch size
                - train.n_epochs: Number of training epochs
                - train.lr: Learning rate
                - train.target_update_freq: K steps for target network sync
                - train.entropy_reg_coef: Alpha for entropy regularization
                - train.curriculum_epochs: Steps for curriculum training
                - train.num_workers: Number of data loading workers
                - train.grad_clip: Gradient clipping value (None to disable)
            task_name: Name of the task
            device: Device to use for training
        """
        self.model = model.to(device)
        self.cfg = cfg
        self.dataset = dataset
        self.boundary_dataset = boundary_dataset
        self.eval_dataset = eval_dataset
        self.device = device

        # Training hyperparameters
        self.batch_size = cfg.train.get("batch_size", 32)
        self.n_epochs = cfg.train.get("n_epochs", 100)
        self.lr = cfg.train.get("lr", 1e-4)
        self.wd = cfg.train.get("wd", 1e-4)
        self.target_update_freq = cfg.train.get("target_update_freq", 100)
        self.entropy_reg_coef = cfg.train.get("entropy_reg_coef", 0.01)
        self.curriculum_epochs = cfg.train.get("curriculum_epochs", 10)
        self.curriculum_steps = cfg.train.get("curriculum_steps", 10)
        self.grad_clip = cfg.train.get("grad_clip", 1.0)
        self.num_workers = cfg.train.get("num_workers", 4)
        self.context_window = cfg.model.get("context_window", 1)
        self.random_decay = cfg.train.get("random_decay", 0.95)
        self.random_decay_noise = cfg.train.get("random_decay_noise", 0.95)

        # Target reward noise parameters (for evaluation data augmentation)
        self.target_noise_mean = cfg.train.get("target_noise_mean", 0.0)
        self.target_noise_std = cfg.train.get("target_noise_std", 0.01)
        self.tau = cfg.train.tau

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        # Training state
        self.current_step = 0
        self.best_loss = float('inf')

        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(self.dataset),
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )
        if self.boundary_dataset is not None:
            self.boundary_dataloader = DataLoader(
                self.boundary_dataset,
                batch_size=self.batch_size,
                sampler=RandomSampler(self.boundary_dataset),
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
            )
        else:
            self.boundary_dataloader = None
        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                sampler=RandomSampler(self.eval_dataset),
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
            )
        # Initialize network with curriculum learning
        curriculum_first = cfg.train.get("curriculum_first", False)
        self.cl_entropy_reg_coef = cfg.train.get("cl_entropy_reg_coef", self.entropy_reg_coef)
        if curriculum_first:
            self._train_curriculum()

        # Create target network (f_target)
        self.target_model = copy.deepcopy(model).to(device)
        # inject_noise_to_target = cfg.train.get("inject_noise_to_target", False)
        # if inject_noise_to_target:
        #     for param in self.target_model.parameters():
        #         param.data += 0.1 * torch.randn_like(param.data)
        # self.target_entropy_reg_coef = cfg.train.get("target_entropy_reg_coef", 0.1)

        # Freeze target network parameters
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.target_model.eval()
        
        # TD Error Filtering parameters
        self.enable_td_filter = cfg.train.get('enable_td_filter', False)
        self.td_error_threshold = cfg.train.get('td_error_threshold', 0.05)
        self.td_error_max_weight = cfg.train.get('td_error_max_weight', 3.0)

        print(f"TD Error Filtering: enabled={self.enable_td_filter}, ")
        print(f"  td_error_threshold={self.td_error_threshold}, td_error_max_weight={self.td_error_max_weight}")
        
        # Statistics tracking
        self.stats = {
            'td_filter_keep_ratio': 0.0,
            'td_error_mean': 0.0,
            'td_error_std': 0.0,
        }

    def train(self, eval_every=10, save_every=2, save_dir=None, use_curriculum=False, start_epoch=0):
        """
        Main training loop.

        Args:
            eval_every: Evaluate every N epochs
            save_every: Save checkpoint every N epochs
            save_dir: Directory to save checkpoints
            use_curriculum: Whether to use curriculum training (None = auto-detect)
            start_epoch: Starting epoch number (for resuming training)

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'eval_loss': [],
            'epoch': [],
        }

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting training for {self.n_epochs} epochs")
        print(f"Batch size: {self.batch_size}, LR: {self.lr}")
        print(f"Target update freq: {self.target_update_freq}")
        print(f"Entropy reg coef: {self.entropy_reg_coef}")
        print(f"Curriculum epochs: {self.curriculum_epochs}")

        for epoch in range(start_epoch, self.n_epochs):
            # Train
            train_metrics = self.train_epoch(epoch, use_curriculum)
            train_loss = train_metrics['total_loss']

            # Log training metrics
            print(
                f"[Epoch {epoch:3d}] "
                f"Loss: {train_loss:.4f} "
                f"(MSE: {train_metrics['mse_loss']:.4f}, "
                f"Entropy: {train_metrics['entropy']:.4f}) "
                f"Entropy Reg Coef: {self.entropy_reg_coef:.5f} "
                f"Noise: {self.target_noise_std:.5f} "
            )

            # Evaluate
            if eval_every > 0 and epoch % eval_every == 0 and self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                eval_loss = eval_metrics['total_loss']

                print(
                    f"  [Eval] Loss: {eval_loss:.4f} "
                    f"(MSE: {eval_metrics['mse_loss']:.4f}, "
                    f"MAE: {eval_metrics['mae']:.4f}, "
                    f"Pred: {eval_metrics['pred_mean']:.3f}±{eval_metrics['pred_std']:.3f})"
                )

                history['eval_loss'].append(eval_loss)
                history['epoch'].append(epoch)

                # Save best model
                if save_dir and eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save(save_dir / "best_model.pth")
                    print(f"  [Eval] Saved best model (loss: {self.best_loss:.4f})")

            history['train_loss'].append(train_loss)
            
            if epoch % 10 == 0:  # increase save frequency every 10 epochs to save storage
                save_every = min(save_every + 1, 5)

            # Save checkpoint
            if save_dir and epoch % save_every == 0:
                self.save(save_dir / f"checkpoint_epoch_{epoch}.pth")
                print(f"  [Checkpoint] Saved checkpoint at epoch {epoch} at {save_dir / f'checkpoint_epoch_{epoch}.pth'}")

            self.reduce_random()

        # Final save
        if save_dir:
            self.save(save_dir / "final_model.pth")
            print(f"  [Checkpoint] Saved final model at {save_dir / 'final_model.pth'}")

        print(f"Training completed!")
        print(f"Best eval loss: {self.best_loss:.4f}")

        return history

    def train_epoch(self, epoch, use_curriculum=None):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            use_curriculum: Whether to use curriculum training for this epoch

        Returns:
            Dictionary with average metrics
        """
        self.model.train()

        total_losses = defaultdict(float)
        num_batches = 0

        if use_curriculum is None:
            use_curriculum = epoch < self.curriculum_epochs

        if use_curriculum:
            assert self.boundary_dataloader is not None, "Boundary dataloader is required for curriculum training."
            dataloader = self.boundary_dataloader
            print(f"Using curriculum training for epoch {epoch}.")
        else:
            dataloader = self.dataloader

        for batch_idx, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader)):
            # Move batch to device
            batch = map_tensor_to_device(batch, device=self.device)

            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.compute_loss(batch, use_curriculum=use_curriculum)
            loss = loss_dict['total_loss']

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

            self.optimizer.step()

            # Accumulate metrics
            for key, value in loss_dict.items():
                if 'td_' not in key:
                    total_losses[key] += value.item()
                else:
                    total_losses[key] += value
            num_batches += 1

            # Update target network every K steps
            if not use_curriculum:
                self.current_step += 1
                if self.current_step % self.target_update_freq == 0:
                    self._update_target_network(tau=self.tau)

        # Compute averages
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        if self.enable_td_filter and not use_curriculum:
            print(f"  [TD Filter] Keep ratio: {avg_losses.get('td_keep_ratio', 0):.3f}, "
                  f"TD error: {avg_losses.get('td_error_mean', 0):.4f}")

        return avg_losses

    def evaluate(self, eval_dataloader=None):
        """
        Evaluate the model.

        Args:
            eval_dataloader: Optional custom dataloader for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        if eval_dataloader is None:
            eval_dataloader = self.eval_dataloader
        assert eval_dataloader is not None, "Evaluation dataloader is not provided."

        total_losses = defaultdict(float)
        total_preds = []
        total_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader)):
                # Assume we use boundary dataset for evaluation
                batch = map_tensor_to_device(batch, device=self.device)
                pred_rewards = self.model(batch).squeeze(-1)  # (B,) 
                target_rewards = batch['rewards'][:, -1]  # (B,)

                # Add Gaussian noise to target rewards for data augmentation
                if self.target_noise_std > 0:
                    noise = torch.randn_like(target_rewards) * self.target_noise_std + self.target_noise_mean
                    target_rewards = target_rewards + noise
                    # Clip to [0, 1] range
                    target_rewards = torch.clamp(target_rewards, 0.0, 1.0)

                # Compute loss
                mse_loss = F.mse_loss(pred_rewards, target_rewards)
                entropy = self._compute_entropy(pred_rewards).mean()
                entropy_reg = -self.entropy_reg_coef * entropy
                total_loss = mse_loss + entropy_reg

                # Accumulate
                total_losses['total_loss'] += total_loss.item()
                total_losses['mse_loss'] += mse_loss.item()
                total_losses['entropy'] += entropy.item()

                total_preds.append(pred_rewards.cpu().numpy())
                total_targets.append(target_rewards.cpu().numpy())

                num_batches += 1

        # Compute averages
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        # Concatenate predictions
        all_preds = np.concatenate(total_preds)
        all_targets = np.concatenate(total_targets)

        # Additional metrics
        avg_losses['pred_mean'] = float(all_preds.mean())
        avg_losses['pred_std'] = float(all_preds.std())
        avg_losses['target_mean'] = float(all_targets.mean())
        avg_losses['mae'] = float(np.abs(all_preds - all_targets).mean())

        return avg_losses

    def save(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.modules_to_save.state_dict(),
            'target_model_state_dict': self.target_model.modules_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path, load_optimizer=True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.modules_to_save.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.modules_to_save.load_state_dict(checkpoint['target_model_state_dict'])

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_step = checkpoint['current_step']
        self.best_loss = checkpoint['best_loss']

        print(f"Loaded checkpoint from {path}")

    def compute_loss(self, batch, use_curriculum=False):
        """
        Compute the loss for a single batch.

        L = MSE(f(H_{1:t}), y_t) - alpha * H(r_t)

        where y_t = r_T if t=T else f_target(H_{1:t+1})

        Args:
            batch: Data batch
            use_curriculum: Whether to use curriculum (None = auto-detect)

        Returns:
            Dictionary with loss components
        """
        if use_curriculum:
            pred_rewards = self.model(batch).squeeze(-1)  # (B,)
            target_rewards = batch['rewards'][:, -1]  # (B,)
        else:
            # Prepare data for TD learning
            # current_batch, batch_targets = self._prepare_batch_for_td(batch)
            current_batch, batch_targets = batch, batch['target_model']
            # Forward pass through main network
            pred_rewards = self.model(current_batch).squeeze(-1)  # (B,) 
            # Forward pass through target network
            with torch.no_grad():
                if 'task_emb' in current_batch:
                    batch_targets['task_emb'] = current_batch['task_emb']
                if 'task_prompt' in current_batch:
                    batch_targets['task_prompt'] = current_batch['task_prompt']
                target_rewards = self.target_model(batch_targets).squeeze(-1)  # (B,) 
                actual_rewards = batch['rewards'][:, -1]  # (B,)
                is_terminal = batch['dones'][:, -1] # (B,)
                # Use terminal reward if episode ended
                target_rewards = torch.where(is_terminal, actual_rewards, target_rewards)
            
        # Add Gaussian noise to target rewards for data augmentation
        if self.target_noise_std > 0:
            noise = torch.randn_like(target_rewards) * self.target_noise_std + self.target_noise_mean
            target_rewards = target_rewards + noise
            # Clip to [0, 1] range
            target_rewards = torch.clamp(target_rewards, 0.0, 1.0)

        # ===== TD Error Filtering (只对高TD误差样本更新) =====
        # Compute TD errors: |pred[t] - target|
        td_errors = torch.abs(pred_rewards - target_rewards)  # (B,)

        # Filter samples based on TD error
        if self.enable_td_filter and not use_curriculum:
            # Soft filtering: use TD errors as sample weights
            # Higher TD error = higher weight
            weights = torch.clamp(td_errors / self.td_error_threshold, 0.0, self.td_error_max_weight)  # (B,)

            # Use weighted MSE loss
            mse_loss = F.mse_loss(pred_rewards, target_rewards, reduction='none')  # (B,)
            mse_loss = (mse_loss * weights).mean()

            # Track statistics
            self.stats['td_filter_keep_ratio'] = weights.mean().item()
            self.stats['td_error_mean'] = td_errors.mean().item()
            self.stats['td_error_std'] = td_errors.std().item()
        else:
            # Standard MSE loss (no filtering)
            mse_loss = F.mse_loss(pred_rewards, target_rewards)

            self.stats['td_filter_keep_ratio'] = 1.0
            self.stats['td_error_mean'] = td_errors.mean().item()
            self.stats['td_error_std'] = td_errors.std().item()

        # MSE loss
        mse_loss = F.mse_loss(pred_rewards, target_rewards)

        # Entropy regularization (to prevent extreme predictions)
        entropy = self._compute_entropy(pred_rewards).mean()
        entropy_reg = -self.entropy_reg_coef * entropy

        # Total loss
        total_loss = mse_loss + entropy_reg

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'entropy': entropy,
            'entropy_reg': entropy_reg,
            'td_keep_ratio': self.stats['td_filter_keep_ratio'],
            'td_error_mean': self.stats['td_error_mean'],
        }

    def reduce_random(self):
        """Decay randomness in the model prediction."""
        self.entropy_reg_coef = max(1e-4, self.entropy_reg_coef * self.random_decay)
        if self.target_noise_std > 0:
            self.target_noise_std = max(5e-3, self.target_noise_std * self.random_decay_noise)

    def _train_target_curriculum(self):
        """
        Train the target network using curriculum learning on boundary dataset.

        Pre-trains the target network for curriculum_steps steps using the boundary dataset.
        This helps initialize the target network with better predictions before the main training loop.
        """
        if self.boundary_dataloader is None:
            print("No boundary dataset provided, skipping target curriculum training.")
            return

        print(f"Pre-training target network for {self.curriculum_steps} steps...")

        # Create a temporary optimizer for the target network
        target_optimizer = torch.optim.AdamW(
            self.target_model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        # Set target network to training mode
        self.target_model.train()

        total_losses = defaultdict(float)
        num_batches = 0

        # Create an iterator for the dataloader
        data_iter = iter(self.boundary_dataloader)

        for step_idx in tqdm(range(self.curriculum_steps), desc="Target Curriculum Training"):
            try:
                batch = next(data_iter)
            except StopIteration:
                # If we exhaust the dataloader, create a new iterator
                data_iter = iter(self.boundary_dataloader)
                batch = next(data_iter)

            # Move batch to device
            batch = map_tensor_to_device(batch, device=self.device)

            # Forward pass: predict rewards
            target_optimizer.zero_grad()
            pred_rewards = self.target_model(batch).squeeze(-1)  # (B,)
            target_rewards = batch['rewards'][:, -1]  # (B,)

            # Add Gaussian noise to target rewards for data augmentation
            if self.target_noise_std > 0:
                noise = torch.randn_like(target_rewards) * self.target_noise_std + self.target_noise_mean
                target_rewards_noisy = target_rewards + noise
                # Clip to [0, 1] range
                target_rewards_noisy = torch.clamp(target_rewards_noisy, 0.0, 1.0)
            else:
                target_rewards_noisy = target_rewards

            # Compute MSE loss
            mse_loss = F.mse_loss(pred_rewards, target_rewards_noisy)

            # Entropy regularization
            entropy = self._compute_entropy(pred_rewards).mean()
            entropy_reg = -self.target_entropy_reg_coef * entropy

            # Total loss
            total_loss = mse_loss + entropy_reg

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.target_model.parameters(),
                    self.grad_clip
                )

            target_optimizer.step()

            # Accumulate metrics
            total_losses['total_loss'] += total_loss.item()
            total_losses['mse_loss'] += mse_loss.item()
            total_losses['entropy'] += entropy.item()
            num_batches += 1

        # Compute averages
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        print(f"Target curriculum training completed. "
              f"Loss: {avg_losses['total_loss']:.4f} "
              f"(MSE: {avg_losses['mse_loss']:.4f}, Entropy: {avg_losses['entropy']:.4f})")

    def _train_curriculum(self):
        """
        Train the network using curriculum learning on boundary dataset.

        Pre-trains the network for curriculum_steps steps using the boundary dataset.
        This helps initialize the network with better predictions before the main training loop.
        """
        if self.boundary_dataloader is None:
            print("No boundary dataset provided, skipping curriculum training.")
            return

        print(f"Pre-training network for {self.curriculum_steps} steps...")

        # Create a temporary optimizer for the network
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        # Set network to training mode
        self.model.train()

        total_losses = defaultdict(float)
        num_batches = 0

        # Create an iterator for the dataloader
        data_iter = iter(self.boundary_dataloader)

        for step_idx in tqdm(range(self.curriculum_steps), desc="Curriculum Training"):
            try:
                batch = next(data_iter)
            except StopIteration:
                # If we exhaust the dataloader, create a new iterator
                data_iter = iter(self.boundary_dataloader)
                batch = next(data_iter)

            # Move batch to device
            batch = map_tensor_to_device(batch, device=self.device)

            # Forward pass: predict rewards
            optimizer.zero_grad()
            pred_rewards = self.model(batch).squeeze(-1)  # (B,)
            target_rewards = batch['rewards'][:, -1]  # (B,)

            # Add Gaussian noise to target rewards for data augmentation
            if self.target_noise_std > 0:
                noise = torch.randn_like(target_rewards) * self.target_noise_std + self.target_noise_mean
                target_rewards_noisy = target_rewards + noise
                # Clip to [0, 1] range
                target_rewards_noisy = torch.clamp(target_rewards_noisy, 0.0, 1.0)
            else:
                target_rewards_noisy = target_rewards

            # Compute MSE loss
            mse_loss = F.mse_loss(pred_rewards, target_rewards_noisy)

            # Entropy regularization
            entropy = self._compute_entropy(pred_rewards).mean()
            entropy_reg = -self.cl_entropy_reg_coef * entropy

            # Total loss
            total_loss = mse_loss + entropy_reg

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

            optimizer.step()

            # Accumulate metrics
            total_losses['total_loss'] += total_loss.item()
            total_losses['mse_loss'] += mse_loss.item()
            total_losses['entropy'] += entropy.item()
            num_batches += 1

        # Compute averages
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        print(f"Curriculum training completed. "
              f"Loss: {avg_losses['total_loss']:.4f} "
              f"(MSE: {avg_losses['mse_loss']:.4f}, Entropy: {avg_losses['entropy']:.4f})")

    def _compute_entropy(self, rewards):
        """
        Compute entropy of reward predictions for regularization.

        H(r_t) = -[r_t * log(r_t) + (1 - r_t) * log(1 - r_t)]

        Args:
            rewards: Tensor of shape (B,) with values in [0, 1]

        Returns:
            Entropy tensor of shape (B,)
        """
        eps = 1e-8
        # Clip rewards to avoid log(0)
        rewards_clipped = torch.clamp(rewards, eps, 1 - eps)
        entropy = -(rewards_clipped * torch.log(rewards_clipped) +
                   (1 - rewards_clipped) * torch.log(1 - rewards_clipped))
        return entropy

    def _prepare_batch_for_td(self, batch):
        """
        Prepare batch for TD(0) style training.

        For each timestep t, we need:
        - f(H_{1:t}) prediction
        - Target y_t = r_T if t=T else f_target(H_{1:t+1})

        This requires shifting the data to get next-step observations.

        Args:
            batch: Data batch with observations and actions
                   Expected format:
                   - obs: dict with observation modalities
                   - actions: (B, T, action_dim)
                   - rewards: (B, T) optional terminal rewards

        Returns:
            Tuple of (current_batch, next_batch, is_terminal_mask, terminal_rewards)
        """
        actions = batch['actions']  # (B, T, action_dim)
        batch_size, seq_len, action_dim = actions.shape

        # 1. Create batch for target network (all timesteps t)
        batch_target = {**batch}

        # 2. Create batch for current model (timesteps [t - self.context_window:-1])
        current_batch = {}
        for key, value in batch.items():
            if key == 'task_emb':
                current_batch[key] = value
            elif isinstance(value, torch.Tensor) and value.ndim > 1:
                current_batch[key] = value[:, :-1]
            elif isinstance(value, dict):
                # Handle nested dict (e.g., obs)
                current_batch[key] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor) and v.ndim > 1:
                        current_batch[key][k] = v[:, :-1]
                    else:
                        current_batch[key][k] = v
            else:
                current_batch[key] = value

        return current_batch, batch_target

    def _update_target_network(self, tau: float = 0.1):
        """Synchronize target network with main network."""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
