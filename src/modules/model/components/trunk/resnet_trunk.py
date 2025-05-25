import torch
from ..registry import ComponentRegistry
from ...nn.architectures import ResNet


@ComponentRegistry.register(component_type='trunk_neural', architecture='resnet')
class ResNetTrunk(torch.nn.Module):
    """ResNet architecture for vanilla/phase2 training"""

    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int, activation: str):
        super().__init__()
        self.net = ResNet(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation
        )

    def forward(self, x):
        return self.net(x)
