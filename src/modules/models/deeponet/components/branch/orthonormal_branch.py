import torch
from src.modules.models.deeponet.components.registry import ComponentRegistry

@ComponentRegistry.register(component_type='orthonormal_branch', architecture='pretrained')
class OrthonormalBranch(torch.nn.Module):
    """
    Branch component for Phase 2.
    Computes M_pred = branch(x) @ R.T and standardizes the output shape.
    """
    def __init__(self, branch: torch.nn.Module, R: torch.Tensor, num_channels: int, is_shared_branch: bool):
        super().__init__()
        self.branch = branch
        self.register_buffer('R', R)
        self.num_channels = num_channels
        self.is_shared_branch = is_shared_branch

    def __str__(self):
        return f"OrthonormalBranch(Branch={self.branch}, R_matrix={self.R.shape})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A_pred = self.branch(x)
        RT = self.R.T

        if self.is_shared_branch:
            B, P = A_pred.shape # A_pred: (B, P)
            C = self.num_channels
            M_pred = torch.empty((B, C, P), device=A_pred.device, dtype=A_pred.dtype)
            for c in range(C):
                rows = slice(c * P, (c + 1) * P)
                Rc_T = RT[rows, rows]
                M_pred[:, c, :] = A_pred @ Rc_T
            return M_pred # M_pred: (B, C, P)
        else:
            B, D = A_pred.shape # A_pred: (B, C*P)
            P = D // self.num_channels
            A_pred_reshaped = A_pred.view(B, self.num_channels, P)

            if RT.shape[0] == D: # Block-diagonal R for SplitOutputs
                M_pred = torch.empty_like(A_pred_reshaped)
                for c in range(self.num_channels):
                    rows = slice(c * P, (c + 1) * P)
                    Rc_T = RT[rows, rows]
                    M_pred[:, c, :] = A_pred_reshaped[:, c, :] @ Rc_T
            else: # Single R for SharedTrunk
                M_pred = A_pred_reshaped @ RT
            return M_pred # M_pred: (B, C, P)