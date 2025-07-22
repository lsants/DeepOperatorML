import torch
import numpy
from typing import Iterable
from ..registry import ComponentRegistry


@ComponentRegistry.register(component_type='matrix_branch', architecture="trainable_matrix")
class MatrixBranch(torch.nn.Module):
    """For two-step phase 1 training (trainable matrix)"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(output_dim, input_dim))
        torch.nn.init.xavier_uniform_(self.weights)
        # torch.nn.init.kaiming_uniform_(self.weights)

    def __str__(self):
        return f"MatrixBranch(input_dim={self.weights.shape[1]}, output_dim={self.weights.shape[0]})"

    def forward(self, index: numpy.ndarray) -> torch.Tensor: # Need to figure out how to correctl index the vector to be multiplied by the sample.
        coefficients_matrix = self.weights.T 
        batch_coefficients = coefficients_matrix[index.flatten()]
        return batch_coefficients
