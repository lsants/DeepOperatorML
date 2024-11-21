import torch
from .deeponet import DeepONet

class PODDeepONet(DeepONet):
    def __init__(self, branch_config, trunk_config, n_outputs):
        super().__init__(branch_config, trunk_config)
        self.register_buffer('POD_basis_initialized', torch.tensor(False))
        self.basis = None

    def forward(self, xb):
        outputs = ()
        branch_out = self.branch_network(xb)
        for basis in self.basis:
            outputs += torch.matmul(basis, torch.transpose(branch_out, 0, 1)),
        return outputs
    
    def get_basis(self, basis):
        self.basis = basis
        self.POD_basis_initialized.copy_(torch.tensor(True, device=basis.device, dtype=torch.bool))