import torch
from .base_trunk import BaseTrunk

class PretrainedTrunk(BaseTrunk, torch.nn.Module):
    def __init__(self, trained_tensor: torch.Tensor):
        """
        A trunk that is represented by a pre-trained Tensor(e.g., trunk network was trained separetely).
        """
        super().__init__()
        self.register_buffer("trained_tensor", trained_tensor)

    def __str__(self):
        input_size, output_size = self.trained_tensor.shape
        return f"Fixed trunk\n({input_size}, {output_size})"
    
    def forward(self) -> torch.Tensor:
        return self.trained_tensor

    def get_basis(self) -> torch.Tensor:
        return self.trained_tensor