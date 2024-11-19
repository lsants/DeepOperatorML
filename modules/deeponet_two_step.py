import torch
from .deeponet import DeepONet

class DeepONetTwoStep(DeepONet):
    def __init__(self, branch_config, trunk_config, A_dim):
        super().__init__(branch_config, trunk_config)
        num_matrices = A_dim[0]
        trainable_matrix_shape = A_dim[1 : ]

        self.A_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(*trainable_matrix_shape))
             for _ in range(num_matrices)])
        
        for A in self.A_list:
            torch.nn.init.xavier_uniform_(A)

        self.R = None
        self.training_phase = 'both'

    def forward(self, xb=None, xt=None):
        if self.training_phase == 'trunk':
            trunk_out = self.trunk_network(xt)
            basis_real = trunk_out @ self.A_list[0]
            basis_imag = trunk_out @ self.A_list[1]
            return basis_real, basis_imag
        
        elif self.training_phase == 'branch':
            branch_out = self.branch_network(xb)
            print(branch_out.shape, "<----- branch output shape before transposing")
            if self.R is None:
                raise ValueError("Basis functions have not been computed. Train trunk first.")
            return branch_out
        else:
            branch_out = self.branch_network(xb)
            trunk_out = self.trunk_network(xt)
            num_basis = trunk_out.shape[1]
            branch_real_out = branch_out[ : , : num_basis]
            branch_imag_out = branch_out[ : , num_basis : ]

            real_out = torch.matmul(trunk_out, torch.transpose(branch_real_out, 0, 1))
            imag_out = torch.matmul(trunk_out, torch.transpose(branch_imag_out, 0, 1))

            return real_out, imag_out
    
    def set_training_phase(self, phase):
        self.training_phase = phase
    
    def freeze_A(self):
        for A in self.A_list:
            A.requires_grad = False

    def unfreeze_A(self):
        for A in self.A_list:
            A.requires_grad = True