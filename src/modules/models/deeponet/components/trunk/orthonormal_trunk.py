import torch
from src.modules.models.deeponet.components.registry import ComponentRegistry

@ComponentRegistry.register(component_type='orthonormal_trunk', architecture="pretrained")
class OrthonormalTrunk(torch.nn.Module):
    """Orthonormalized trunk (SVD/QR) for phase2 inference"""

    def __init__(self, trunk: torch.nn.Module, T: torch.Tensor, num_channels: int, is_shared_trunk: bool):
        super().__init__()
        self.trunk = trunk
        self.register_buffer(name='T', tensor=T)
        self.num_channels = num_channels
        self.is_shared_trunk = is_shared_trunk
        self.requires_grad_(False)

    def __str__(self):
        return f"OrthonormalTrunk(trunk={self.trunk}, T_matrix={self.T.shape})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            phi = self.trunk(x)
            phi_ortho = phi @ self.T
            if self.is_shared_trunk:
                return phi_ortho
            else:
                T_dim, _ = phi_ortho.shape
                return phi_ortho.view(T_dim, self.num_channels, -1)