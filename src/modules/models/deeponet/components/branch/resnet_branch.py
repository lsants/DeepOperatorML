import torch
from typing import Callable, Optional
from src.modules.models.deeponet.components.registry import ComponentRegistry
from src.modules.models.tools.architectures import ResNet

@ComponentRegistry.register(component_type='neural_branch', architecture='resnet')
class ResNetBranch(torch.nn.Module):
    """ResNet architecture for vanilla/phase2 training"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout_rates: Optional[list[float]] = None,
        batch_normalization: Optional[list[bool]] = None,
        layer_normalization: Optional[list[bool]] = None,
    ) -> None:
        super().__init__()
        self.net = ResNet(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation,
            dropout_rates=dropout_rates,
            batch_normalization=batch_normalization,
            layer_normalization=layer_normalization,
        )

    def forward(self, x):
        return self.net(x)
