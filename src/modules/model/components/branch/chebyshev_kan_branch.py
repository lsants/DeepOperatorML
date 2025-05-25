import torch
from platform import architecture
from ..registry import ComponentRegistry
from ...nn.architectures import ChebyshevKAN


@ComponentRegistry.register(component_type='branch_neural', architecture='chebyshev_kan')
class ChebyshevKANBranch(torch.nn.Module):
    """KAN architecture implementation"""

    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int, degree: int):
        super().__init__()
        self.net = ChebyshevKAN(
            input_dim=input_dim,
            layers=hidden_layers,
            output_dim=output_dim,  # Assuming output_dim is same as input_dim for KAN
            degree=degree
        )

    def forward(self, x):
        return self.net(x)
