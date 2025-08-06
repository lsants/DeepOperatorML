import torch
from src.modules.models.deeponet.components.registry import ComponentRegistry
from src.modules.models.tools.architectures import ChebyshevKAN

@ComponentRegistry.register(component_type='branch_neural', architecture='chebyshev_kan')
class ChebyshevKANBranch(torch.nn.Module):
    """KAN architecture implementation"""

    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int, degree: int):
        super().__init__()
        self.net = ChebyshevKAN(
            input_dim=input_dim,
            layers=hidden_layers,
            output_dim=output_dim,
            degree=degree
        )

    def forward(self, x):
        return self.net(x)
