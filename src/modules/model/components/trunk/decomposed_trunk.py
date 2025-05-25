import torch
from ..registry import ComponentRegistry


@ComponentRegistry.register(component_type='decomposed')
class DecomposedTrunk(torch.nn.Module):
    """Fixed decomposed trunk (SVD/QR) for phase2 inference"""

    def __init__(self, decomposed_tensor: torch.Tensor):
        super().__init__()
        # [input_dim, reduced_dim]
        self.register_buffer('decomposed_tensor', decomposed_tensor)

    def forward(self, x):
        return x @ self.decomposed_tensor  # Projection
