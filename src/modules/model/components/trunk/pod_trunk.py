import torch
from ..registry import ComponentRegistry


@ComponentRegistry.register(component_type='pod_trunk', architecture="precomputed")
class PODTrunk(torch.nn.Module):
    """Precomputed POD basis trunk"""

    def __init__(self, pod_basis: torch.Tensor, pod_mean: torch.Tensor):
        super().__init__()
        # [coord_samples, num_modes]
        self.register_buffer('pod_basis', pod_basis) # [coord_samples, C*P]
        self.register_buffer('pod_mean', pod_mean)  # [coord_samples,]

    def forward(self, x):
        return self.pod_basis
