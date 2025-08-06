import torch
from src.modules.models.deeponet.components.registry import ComponentRegistry
from src.modules.models.tools.architectures import ChebyshevKAN


@ComponentRegistry.register(component_type='neural_trunk', architecture='chebyshev_kan')
class ChebyshevKANTrunk(torch.nn.Module):
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
