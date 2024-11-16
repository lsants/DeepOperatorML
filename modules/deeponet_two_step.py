import torch
from .deeponet import DeepONet

class DeepONetTwoStep(DeepONet):
    def __init__(self, branch_config, trunk_config, A_dim):
        super().__init__(branch_config, trunk_config)
        trainable_matrix = torch.nn.Parameter(
            torch.randn(*A_dim)
        )
        self.A_real = trainable_matrix[0]
        self.A_imag = trainable_matrix[1]
        self.R = None
        self.training_phase = 'both'

    def forward(self, xb=None, xt=None):
        if self.training_phase == 'trunk':
            trunk_out = self.trunk_network(xt)
            basis_real = trunk_out @ self.A_real
            basis_imag = trunk_out @ self.A_imag
            return basis_real, basis_imag
        elif self.training_phase == 'branch':
            branch_out = self.branch_network(xb)
            if self.R is None:
                raise ValueError("Basis functions have not been computed. Train trunk first.")
            return branch_out
        else:
            branch_out = self.branch_network(xb)
            trunk_out = self.trunk_network(xt)
            num_basis = trunk_out.shape[1]
            branch_real_out = branch_out[ : , : num_basis]
            branch_imag_out = branch_out[ : , num_basis : ]

            real_out = torch.matmul(branch_real_out, torch.transpose(trunk_out, 0, 1))
            imag_out = torch.matmul(branch_imag_out, torch.transpose(trunk_out, 0, 1))

            return real_out, imag_out
        
    def freeze_branch(self):
        for param in self.branch_network.parameters():
            param.requires_grad = False

    def unfreeze_branch(self):
        for param in self.branch_network.parameters():
            param.requires_grad = True
    
    def freeze_trunk(self):
        for param in self.trunk_network.parameters():
            param.requires_grad = False

    def unfreeze_trunk(self):
        for param in self.trunk_network.parameters():
            param.requires_grad = True
    
    def set_training_phase(self, phase):
        self.training_phase = phase