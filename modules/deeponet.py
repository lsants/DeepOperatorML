import torch
from .mlp import MLP
from .kan import ChebyshevKAN

NETWORK_ARCHITECTURES = {
    'mlp': MLP,
    'kan': ChebyshevKAN,
}

class DeepONet(torch.nn.Module):
    def __init__(self, branch_config, trunk_config):
        super().__init__()
        self.branch_network = self.create_network(branch_config)
        self.trunk_network = self.create_network(trunk_config)

    def create_network(self, config):
        config = config.copy()
        architecture_name = config.pop('architecture').lower()
        try:
            constructor = NETWORK_ARCHITECTURES[architecture_name]
        except KeyError:
            raise ValueError(f"Architecture '{architecture_name}' not implemented.")
        
        return constructor(**config)

    def forward(self, xb, xt):
        branch_out = self.branch_network(xb)
        trunk_out = self.trunk_network(xt)
        num_basis = trunk_out.shape[1]
        branch_real_out = branch_out[ : , : num_basis]
        branch_imag_out = branch_out[ : , num_basis : ]

        real_out = torch.matmul(branch_real_out, torch.transpose(trunk_out, 0, 1))
        imag_out = torch.matmul(branch_imag_out, torch.transpose(trunk_out, 0, 1))

        return real_out, imag_out