import torch.nn as nn


class RobotStateEncoder(nn.Module):
    def __init__(
        self,
        robot_states_dim=8,
        extra_num_layers=0,
        extra_hidden_size=64,
        embedding_size=32,
    ):
        """
        This is a class that maps all extra modality inputs into tokens of the same size
        """
        super().__init__()
        self.embedding_size = embedding_size

        self.robot_states_dim = robot_states_dim


        def generate_proprio_mlp_fn(extra_low_level_feature_dim):
            assert extra_low_level_feature_dim > 0  # we indeed have extra information
            if extra_num_layers > 0:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_hidden_size)]
                for i in range(1, extra_num_layers):
                    layers += [
                        nn.Linear(extra_hidden_size, extra_hidden_size),
                        nn.ReLU(inplace=True),
                    ]
                layers += [nn.Linear(extra_hidden_size, embedding_size)]
            else:
                layers = [nn.Linear(extra_low_level_feature_dim, embedding_size)]

            proprio_mlp = nn.Sequential(*layers)
            return proprio_mlp
    
        self.encoders = generate_proprio_mlp_fn(self.robot_states_dim)

    def forward(self, obs_dict):
        """
        obs_dict: {
            (optional) robot_stats: (B, T, 8),
        }
        map above to a latent vector of shape (B, T, 1, H)
        """
        x = self.encoders(obs_dict["robot_states"])
        x = x.unsqueeze(-2)  # (B, T, 1, H)
        return x