from __future__ import annotations
import torch


class Bias(torch.nn.Module):
    def __init__(self, num_channels: int, precomputed_mean: torch.Tensor | None = None):
        """
        Flexible bias component that can be either:
        - Trainable parameter (standard case)
        - Fixed buffer (for POD/precomputed cases)

        Args:
            num_channels: Size of the bias vector
            precomputed_mean: Mean functions in the case of POD
        """
        super().__init__()
        if precomputed_mean is not None:
            self.register_buffer('bias', precomputed_mean.T)
        else:
            self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def __str__(self):
        return f"Bias(shape={self.bias.shape})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias
