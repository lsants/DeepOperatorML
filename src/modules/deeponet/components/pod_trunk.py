import torch
from .base_trunk import BaseTrunk

class PODTrunk(BaseTrunk):
    def __init__(self, basis: torch.Tensor, mean: torch.Tensor):
        """
        A trunk that is represented by a fixed tensor (e.g., computed basis functions).
        """
        super().__init__()
        self.basis = basis
        self.mean = mean
        self.register_buffer("basis", basis)
        self.register_buffer("mean", mean)

    def __str__(self):
        input_size, output_size = self.basis.shape
        return f"POD trunk\n({input_size}, {output_size})"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.basis

    def get_basis(self) -> torch.Tensor:
        return self.basis