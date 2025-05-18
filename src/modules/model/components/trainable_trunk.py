import torch
import torch.nn as nn
from .base_trunk import BaseTrunk

class TrainableTrunk(BaseTrunk, torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        A trunk that is a trainable neural network.
        """
        super().__init__()
        self.module = module

    def __str__(self):
        def get_layer_sizes(module):
            first_layer = next(iter(module.children()))
            if hasattr(first_layer, 'in_features'):
                input_size = first_layer.in_features
            elif hasattr(first_layer, 'in_channels'):
                input_size = first_layer.in_channels
            else:
                raise ValueError("Could not determine input size of the first layer.")
            last_layer = list(module.children())[-1]
            if hasattr(last_layer, 'out_features'):
                output_size = last_layer.out_features
            elif hasattr(last_layer, 'out_channels'):
                output_size = last_layer.out_channels
            else:
                raise ValueError("Could not determine output size of the last layer.")
            return input_size, output_size
        input_size, output_size = get_layer_sizes(self.module)
        
        return f"Trainable trunk\n({input_size}, {output_size})"

    def forward(self, trunk_input: torch.Tensor) -> torch.Tensor:
        return self.module(trunk_input)

    def get_basis(self) -> torch.Tensor:
        # In a trainable trunk, you might simply compute the basis by a forward pass.
        # Alternatively, additional processing could be applied.
        return self.module.forward(torch.empty(1))  # Dummy call; adjust as needed.