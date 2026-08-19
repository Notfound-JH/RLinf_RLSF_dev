import torch
import torch.nn as nn
import math


class TimestepEncoder(nn.Module):
    def __init__(
        self,
        max_timesteps: int = 1000,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 1,
        base: float = 5000.0,
    ):
        """
        Precomputed sinusoidal timestep embedding + trainable MLP.
        
        Args:
            max_timesteps: Maximum number of timesteps (e.g., 1000).
            embedding_dim: Dimension of sinusoidal embedding (must be even).
            hidden_dim: Hidden size in MLP.
            num_layers: Number of MLP layers.
            base: Frequency base (different from positional encoding).
        """
        super().__init__()
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim
        self.base = base

        # Step 1: Precompute sinusoidal embeddings for all t in [0, max_timesteps)
        self.register_buffer(
            "precomputed_emb",
            self._precompute_embeddings(max_timesteps, embedding_dim, base),
            persistent=False  # not saved in state_dict if you don't want it
        )

        # Step 2: Build MLP (trainable)
        layers = []
        input_dim = embedding_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else embedding_dim
            layers.append(nn.Linear(input_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            input_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _precompute_embeddings(max_timesteps, dim, base):
        """Precompute sinusoidal embeddings for t = 0, 1, ..., max_timesteps-1."""
        assert dim % 2 == 0
        half_dim = dim // 2
        t = torch.arange(max_timesteps, dtype=torch.float32)  # (T,)
        exponent = torch.arange(half_dim, dtype=torch.float32)
        freqs = torch.exp(-exponent * math.log(base) / half_dim)  # (half_dim,)
        args = t[:, None] * freqs[None, :]  # (T, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (T, dim)
        return emb

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Long tensor of shape (N,) with values in [0, max_timesteps)
        Returns:
            Embedded timesteps of shape (N, embedding_dim)
        """
        # Lookup precomputed embeddings
        emb = self.precomputed_emb[timesteps]  # (N, dim)
        # Pass through trainable MLP
        return self.mlp(emb)