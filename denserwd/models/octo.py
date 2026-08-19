import numpy as np
import torch
import torch.nn as nn
from typing import List
from denserwd.octo.model.components.transformer_pt import TransformerPt
from denserwd.octo.model.octo_model_pt import OctoModelPt
from denserwd.models.encoders.image import ResNet50ImageEncoder
from denserwd.models.encoders.language import T5LanguageEncoder
from denserwd.models.encoders.robot import RobotStateEncoder
from denserwd.models.encoders.timestep import TimestepEncoder


class RewardToken(nn.Module):
    """Learnable reward token to be appended to transformer input sequence."""
    
    def __init__(self, token_embedding_dim: int):
        super().__init__()
        self.reward_token = nn.Parameter(torch.randn(1, 1, token_embedding_dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.reward_token.expand(batch_size, -1, -1)  # (B, 1, E)


class OctoReward(nn.Module):
    """Reward model using Octo architecture components.

    Combines:
    - T5-base language encoder
    - ResNet50 image encoder (pretrained)
    - Block transformer from OctoModelPt

    The model predicts rewards for state-action pairs conditioned on text instructions.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.token_embedding_dim = cfg.model.token_embedding_dim
        self.max_ep_len = cfg.model.max_ep_len
        self.device = cfg.device

        # 1. Language encoder (T5-base)
        self.language_encoder = T5LanguageEncoder(
            model_name=cfg.model.language_model,
            finetune=cfg.model.finetune_language,
            device=self.device
        )
        self.language_proj = nn.Linear(
            self.language_encoder.output_dim, self.token_embedding_dim
        )

        # 2. Image encoder (ResNet50)
        self.image_encoder = ResNet50ImageEncoder(
            finetune=cfg.model.finetune_vision,
            device=self.device
        )
        self.image_proj = nn.Linear(
            self.image_encoder.output_dim, self.token_embedding_dim
        )
        self.images_to_encode = cfg.data.obs.modality.rgb + cfg.data.obs.modality.depth

        # 3. Robotic state encoder (MLP)
        self.robot_encoder = RobotStateEncoder(
            robot_states_dim=8,
            extra_num_layers=cfg.model.extra_num_layers,
            extra_hidden_size=cfg.model.extra_hidden_size,
            embedding_size=self.token_embedding_dim
        )
        
        # 4. Timestep encoder
        self.timestep_encoder = TimestepEncoder(
            max_timesteps=self.max_ep_len,
            embedding_dim=self.token_embedding_dim
        )

        # 5. Action encoder (MLP)
        action_dim = cfg.model.action_dim
        self.action_encoder = nn.Linear(action_dim, self.token_embedding_dim)

        # 6. Layer norm
        self.layer_norm = nn.LayerNorm(self.token_embedding_dim)

        # 7. Reward token
        self.reward_token = RewardToken(token_embedding_dim=self.token_embedding_dim)

        # 8. Block transformer (loaded from OctoModelPt if pretrained path provided)
        self.block_transformer = TransformerPt(token_embedding_size=self.token_embedding_dim, **cfg.model.transformer_kwargs)  
        if cfg.model.octo_pretrained_path is not None:
            self._init_block_transformer_from_octo(cfg.model.octo_pretrained_path)

        # 9. Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(self.token_embedding_dim, 1),
            nn.Sigmoid()
        )

        # Collect modules need to save (some encoders are frozen and no need to save)
        self.modules_to_save = nn.ModuleList([
            self.language_proj,
            self.image_proj,
            self.robot_encoder,
            self.timestep_encoder,
            self.action_encoder,
            self.layer_norm,
            self.reward_token,
            self.block_transformer,
            self.reward_head
        ])
    
    def get_task_embs(self, task_descriptions: List[str]):
        with torch.no_grad():
            task_embs_tensors = self.language_encoder(task_descriptions)
            task_embs = []
            for i in range(task_embs_tensors.shape[0]):
                task_embs.append(task_embs_tensors[i])
        return task_embs

    def forward(self, data, train: bool = True):
        # 1. Encode text
        text_encoded = self._language_encode(data)
        # 2. Encode timesteps
        time_embeddings = self._timestep_encode(data["timesteps"])  # (B, T, 1, E)
        # 3. Encode observations and actions
        obs_encoded = self._obs_encode(data)  # (B, T, num_modalities, E)
        action_encoded = self._action_encode(data).unsqueeze(2)  # (B, T, 1, E)
        # 4. Stack observations and actions, add time embeddings
        stacked = torch.cat([obs_encoded, action_encoded], dim=2)  # (B, T, num_modalities+1, E)
        stacked = stacked + time_embeddings  # (B, T, num_modalities+1, E)
        stacked = stacked.view(stacked.shape[0], -1, stacked.shape[-1])  # (B, T*(num_modalities+1), E)
        # 5. Concat text and obs-action encodings
        input_tensors = torch.cat([text_encoded, stacked], dim=1)  # (B, text_tokens+T*(num_modalities+1), E)
        input_tensors = self.layer_norm(input_tensors)
        # 6. Append reward token
        reward_tokens = self.reward_token(stacked.shape[0])  # (B, 1, E)
        input_tensors = torch.cat([input_tensors, reward_tokens], dim=1)  # (B, text_tokens+T*(num_modalities+1)+1, E)
        # 7. Create causal attention mask
        attention_mask = self._create_causal_attention_mask(
            input_tensors.shape[1],
            self.block_transformer.num_attention_heads,
            self.device
        )
        # 8. Forward through block transformer
        hidden_states = self.block_transformer(input_tensors, attention_mask, train)  # (B, seq_len, E)
        # 9. Predict reward from last hidden state
        last_hidden_states = hidden_states[:, -1, :]  # (B, E)
        rewards = self.reward_head(last_hidden_states)  # (B, 1)
        return rewards
    
    def predict(self, data):
        return self.forward(data, train=False)

    def _timestep_encode(self, timesteps):
        timesteps = torch.clip(timesteps, max=self.max_ep_len, min=0)
        timestep_embeddings = self.timestep_encoder(timesteps).unsqueeze(2)
        return timestep_embeddings

    def _obs_encode(self, data):
        # 1. encode robot state
        robot = self.robot_encoder(data["obs"])  # (B, T, num_robot_state, E)

        # 2. encode image
        B, T = robot.shape[:2]
        encoded = [robot]
        for img_name in self.images_to_encode:
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            img_encoded = self.image_encoder(
                x.reshape(B * T, C, H, W),
            ).view(B, T, 1, -1)
            encoded.append(self.image_proj(img_encoded))
        encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
        return encoded

    def _language_encode(self, data):
        text_encoded = self.language_proj(data["task_emb"])
        return text_encoded

    def _action_encode(self, data):
        action_encoded = self.action_encoder(data["actions"])  # (B, T, E)
        return action_encoded

    def _create_causal_attention_mask(
        self,
        input_len: int,
        num_attention_heads: int,
        device: torch.device
    ) -> torch.Tensor:
        """Creates a causal attention mask for the transformer.

        Creates a simple causal mask where each position can only attend to
        itself and previous positions in the sequence.

        Args:
            input_len: Number of input tokens
            num_attention_heads: Number of attention heads
            device: Device to create the mask on

        Returns:
            Boolean attention mask of shape (num_attention_heads, seq_len, seq_len)
            True values indicate positions that should be masked (not attended to)
        """
        # Create causal mask: upper triangular matrix (excluding diagonal) is True
        # For i <= j, mask is False (can attend), for i > j, mask is True (cannot attend)
        causal_mask = torch.triu(
            torch.ones(input_len, input_len, device=device),
            diagonal=1
        ).bool()  # (seq_len, seq_len)

        return causal_mask

    def _init_block_transformer_from_octo(self, pretrained_path: str):
        """Init block transformer from the pretrained OctoModel.

        Loads the first k layers from the pretrained OctoModel's transformer,
        where k is the number of layers in self.block_transformer.

        Args:
            pretrained_path: Path to pretrained Octo model checkpoint
        """
        octo_model = OctoModelPt.load_pretrained_from_jax(pretrained_path)['octo_model']
        pretrained_transformer = octo_model.module.octo_transformer.block_transformer.transformer
        target_transformer = self.block_transformer

        # Get the number of layers to copy
        num_layers_to_copy = min(
            target_transformer.num_layers,
            pretrained_transformer.num_layers
        )

        print(f"Loading {num_layers_to_copy} transformer layers from pretrained Octo model "
              f"(pretrained has {pretrained_transformer.num_layers} layers, "
              f"target has {target_transformer.num_layers} layers)")

        # Copy encoder blocks layer by layer
        for i in range(num_layers_to_copy):
            target_block = target_transformer.encoder_blocks[i]
            pretrained_block = pretrained_transformer.encoder_blocks[i]

            # Copy all parameters from the pretrained block to the target block
            target_state_dict = target_block.state_dict()
            pretrained_state_dict = pretrained_block.state_dict()

            # Load matching parameters
            for name, param in pretrained_state_dict.items():
                if name in target_state_dict and target_state_dict[name].shape == param.shape:
                    target_state_dict[name].copy_(param)
                else:
                    print(f"Warning: Skipping parameter {name} due to shape mismatch or missing key")
        print("Loaded encoder_blocks parameters")

        # Copy layer norm if it exists and shapes match
        if (hasattr(target_transformer, 'layer_norm') and
            hasattr(pretrained_transformer, 'layer_norm')):
            target_ln = target_transformer.layer_norm
            pretrained_ln = pretrained_transformer.layer_norm
            
            # Copy all parameters from the pretrained block to the target block
            target_state_dict = target_ln.state_dict()
            pretrained_state_dict = pretrained_ln.state_dict()

            # Load matching parameters
            for name, param in pretrained_state_dict.items():
                if name in target_state_dict and target_state_dict[name].shape == param.shape:
                    target_state_dict[name].copy_(param)
                else:
                    print(f"Warning: Skipping parameter {name} due to shape mismatch or missing key")
            print("Loaded layer_norm parameters")

        # Copy position embedding if both have it and shapes match
        if (hasattr(target_transformer, 'position_embedding') and
            hasattr(pretrained_transformer, 'position_embedding')):
            target_pe = target_transformer.position_embedding
            pretrained_pe = pretrained_transformer.position_embedding
            
            # Copy all parameters from the pretrained block to the target block
            target_state_dict = target_pe.state_dict()
            pretrained_state_dict = pretrained_pe.state_dict()

            # Load matching parameters
            for name, param in pretrained_state_dict.items():
                if name in target_state_dict and target_state_dict[name].shape == param.shape:
                    target_state_dict[name].copy_(param)
                else:
                    print(f"Warning: Skipping parameter {name} due to shape mismatch or missing key")
            print("Loaded position_embedding parameters")

        print(f"Successfully loaded {num_layers_to_copy} transformer layers from pretrained Octo model.")
        del octo_model
