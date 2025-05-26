import torch
import torch.nn as nn


class Noiser(nn.Module):
    def __init__(self, num_continuous: int):
        super().__init__()
        self.output_logsigma_fn = nn.Linear(num_continuous, num_continuous, bias=True)
        torch.nn.init.zeros_(self.output_logsigma_fn.weight)
        torch.nn.init.zeros_(self.output_logsigma_fn.bias)
        self.output_logsigma_fn.weight.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_logsigma_fn(x)
