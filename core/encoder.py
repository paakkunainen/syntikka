import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
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

        self.latent_dim = latent_dim
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        # Output layer for mu and logvar
        layers.append(nn.Linear(hidden_dim, 2 * latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outs = self.net(x)
        mu = outs[:, : self.latent_dim]
        logvar = outs[:, self.latent_dim :]
        return mu, logvar
