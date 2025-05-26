import torch
import torch.nn as nn
from typing import List


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_continuous: int,
        num_categories: List[int],
        hidden_dim: int = 128,
        dropout: float = 0.5,
        num_layers: int = 2,
        device: str = "auto",
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_continuous = num_continuous
        self.num_categories = num_categories
        output_dim = num_continuous + sum(num_categories)

        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
