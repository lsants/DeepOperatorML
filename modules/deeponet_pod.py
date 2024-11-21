import torch
from .deeponet import DeepONet

class PODDeepONet(DeepONet):
    def __init__(self, branch_config, trunk_config, n_outputs):
        super().__init__(branch_config, trunk_config)
        self.register_buffer('POD_basis_initialized', torch.tensor(False))
        self.register_buffer('mean_functions_initialized', torch.tensor(False))
        self.mean_functions = None
        self.basis = None

    def forward(self, xb):
        outputs = ()
        branch_out = self.branch_network(xb)
        for basis, mean in zip(self.basis, self.mean_functions):
            outputs += (torch.matmul(basis, torch.transpose(branch_out, 0, 1)) + mean),
        return outputs
    
    def get_basis(self, basis):
        self.basis = basis
        self.POD_basis_initialized.copy_(torch.tensor(True, device=basis.device, dtype=torch.bool))
    
    def get_mean_functions(self, means):
        self.mean_functions = means
        self.mean_functions_initialized.copy_(torch.tensor(True, device=means.device, dtype=torch.bool))