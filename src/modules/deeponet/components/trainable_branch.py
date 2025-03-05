import torch
from .base_branch import BaseBranch

class TrainableBranch(BaseBranch,  torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        A branch implemented as a trainable neural network.
        """
        super().__init__()
        self.module = module

    def __str__(self):
        def get_layer_sizes(module):
            children = list(module.children())
            if len(children) == 0:
        # Use the module itself if no children exist.
                target = module
            else:
                target = children[0]
            if hasattr(target, 'in_features'):
                input_size = target.in_features
            elif hasattr(target, 'in_channels'):
                input_size = target.in_channels
            else:
                raise ValueError("Could not determine input size of the first layer.")
            
            if len(children) == 0:
                target = module
            else:
                target = children[-1]
            if hasattr(target, 'out_features'):
                output_size = target.out_features
            elif hasattr(target, 'out_channels'):
                output_size = target.out_channels
            else:
                raise ValueError("Could not determine output size of the module.")
            return input_size, output_size
        input_size, output_size = get_layer_sizes(self.module)
        
        return f"Trainable branch\n({input_size}, {output_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)