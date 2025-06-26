import torch
from ..registry import ComponentRegistry
from ...nn.architectures import MLP


@ComponentRegistry.register(component_type='neural_trunk', architecture='mlp')
class MLPTrunk(torch.nn.Module):
    """Trainable neural network trunk"""

    def __init__(self, hidden_layers: list[int], activation: str, input_dim: int, output_dim: int):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation
        )

    def forward(self, x):
        return self.net(x)
