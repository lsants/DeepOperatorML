import torch
from .config import RescalingConfig


class Rescaler(torch.nn.Module):
    def __init__(self, config: RescalingConfig):
        super().__init__()
        self.scale = config.num_basis_functions ** config.exponent

    def __str__(self) -> str:
        return f"Scaler: {self.scale}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
