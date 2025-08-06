import torch
from src.modules.models.deeponet.components.registry import ComponentRegistry


@ComponentRegistry.register(component_type='pod_trunk', architecture="precomputed")
class PODTrunk(torch.nn.Module):
    """Precomputed POD basis trunk"""

    def __init__(self, pod_basis: torch.Tensor):
        super().__init__()
        # [coord_samples, num_modes]
        self.register_buffer('pod_basis', pod_basis)  # [coord_samples, C*P]

    def __str__(self):
        return f"PODTrunk(pod_basis={self.pod_basis.shape})"

    def forward(self, x):
        return self.pod_basis
