import torch
from ..registry import ComponentRegistry

@ComponentRegistry.register(component_type='orthonormal_trunk', architecture="pretrained")
class OrthonormalTrunk(torch.nn.Module):
    """Orthonormalized trunk (SVD/QR) for phase2 inference"""

    def __init__(self, trunk: torch.nn.Module, T: torch.Tensor):
        super().__init__()
        self.trunk = trunk
        self.register_buffer(name='T', tensor=T)
        self.requires_grad_(False)

    def __str__(self):
        return f"OrthonormalTrunk(trunk={self.trunk}, T_matrix={self.T.shape})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            trunk_out = self.trunk(x)
            return trunk_out @ self.T  # Orthonormalized output
