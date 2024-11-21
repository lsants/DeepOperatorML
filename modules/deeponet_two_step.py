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

        self.register_buffer('Q_initialized', torch.tensor(False))
        self.register_buffer('R_initialized', torch.tensor(False))
        self.register_buffer('T_initialized', torch.tensor(False))

        self.R = None
        self.Q = None
        self.T = None
        self.training_phase = 'both'

    def forward(self, xb=None, xt=None):
        if self.training_phase == 'trunk':
            trunk_out = self.trunk_network(xt)
            basis_real = trunk_out @ self.A_list[0]
            basis_imag = trunk_out @ self.A_list[1]
            
            return basis_real, basis_imag
        
        elif self.training_phase == 'branch':
            if not self.R_initialized:
                raise ValueError("Basis functions have not been computed. Train trunk first.")
            branch_out = self.branch_network(xb)
            return branch_out
        else:
            if not self.Q_initialized or not self.T_initialized:
                raise ValueError("Error in basis function decomposition. Recheck.")

            branch_out = self.branch_network(xb)
            trunk_out = self.Q @ self.T

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
    
    def set_Q(self, Q_matrix):
        self.Q = Q_matrix
        self.Q_initialized.copy_(torch.tensor(True, device=Q_matrix.device, dtype=torch.bool))

    def set_R(self, R_matrix):
        self.R = R_matrix
        self.R_initialized.copy_(torch.tensor(True, device=R_matrix.device, dtype=torch.bool))

    def set_T(self, T_matrix):
        self.T = T_matrix
        self.T_initialized.copy_(torch.tensor(True, device=T_matrix.device, dtype=torch.bool))