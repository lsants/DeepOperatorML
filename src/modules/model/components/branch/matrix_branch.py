import torch
from ..registry import ComponentRegistry


@ComponentRegistry.register(component_type='matrix_branch', architecture="trainable_matrix")
class MatrixBranch(torch.nn.Module):
    """For two-step phase 1 training (trainable matrix)"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x):
        return x @ self.weights.T
