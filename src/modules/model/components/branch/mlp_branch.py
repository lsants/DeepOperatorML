import torch
from ..registry import ComponentRegistry
from ...nn.architectures import MLP

@ComponentRegistry.register(component_type='neural_branch', architecture='mlp')
class MLPBranch(torch.nn.Module):
    """Standard MLP implementation"""

    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int, activation: str):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation
        )

    def forward(self, x):
        return self.net(x)
