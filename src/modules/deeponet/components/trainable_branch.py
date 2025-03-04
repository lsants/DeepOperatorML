import torch
import torch.nn as nn
from base_branch import BaseBranch

class TrainableBranch(BaseBranch):
    def __init__(self, module: nn.Module):
        """
        A branch implemented as a trainable neural network.
        """
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
        
        return f"Trainable branch\n({input_size}, {output_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)