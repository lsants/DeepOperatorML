import torch
from ..registry import ComponentRegistry

@ComponentRegistry.register(component_type='pod_trunk', architecture="precomputed")
class PODTrunk(torch.nn.Module):
    """Precomputed POD basis trunk"""

    def __init__(self, modes: torch.Tensor, input_dim: int):
        super().__init__()
        self.register_buffer('modes', modes)  # [input_dim, num_modes]

    def forward(self, x):
        return x @ self.modes
