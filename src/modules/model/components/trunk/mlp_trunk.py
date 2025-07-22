from __future__ import annotations
import torch
from typing import Callable, Iterable, Optional
from ..registry import ComponentRegistry
from ...nn.architectures import MLP

@ComponentRegistry.register(component_type='neural_trunk', architecture='mlp')
class MLPTrunk(torch.nn.Module):
    """Trainable neural network trunk"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout_rates: Optional[list[float]] = None,
        batch_normalization: Optional[list[bool]] = None,
        layer_normalization: Optional[list[bool]] = None,
     ):
        super().__init__()
        self.net = MLP(
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

