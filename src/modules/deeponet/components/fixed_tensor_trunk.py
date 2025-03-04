import torch
import torch.nn as nn
from base_trunk import BaseTrunk

class FixedTensorTrunk(BaseTrunk):
    def __init__(self, fixed_tensor: torch.Tensor):
        """
        A trunk that is represented by a fixed tensor (e.g., computed basis functions).
        """
        self.fixed_tensor = fixed_tensor

    def __str__(self):
        input_size, output_size = self.fixed_tensor.shape
        return f"Fixed trunk\n({input_size}, {output_size})"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fixed_tensor

    def get_basis(self) -> torch.Tensor:
        return self.fixed_tensor