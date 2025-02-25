import torch 
from .output_handling_base import OutputHandlingStrategy

class MultipleTrunksSingleBranchStrategy(OutputHandlingStrategy):
    """Using multiple trunk networks to map the operator's input space.
       This allows learning of significantly different basis functions for each outputs.
       Here, it is considered that the input function mapping is learnable with a single network.
    """

    def __init__(self):
        self.branch_output_dim = None
        self.trunk_output_dim = None

    def get_basis_config(self):
        return {'type': 'multiple'}
    
    def configure_networks(self, model, branch_config, trunk_config, **kwargs):
        
        pod_basis = getattr(model, 'pod_basis', None)
        n_basis_functions = model.n_basis_functions

        if pod_basis is not None:
            if pod_basis.shape[0] < 2:
                raise ValueError("MultipleTrunksSingleBranch strategy expects multiple sets of basis functions with shape (n_basis_sets, n_features, n_modes), with n_basis_sets > 1.")
            n_basis_functions = pod_basis.shape[1]
            model.n_basis_functions = n_basis_functions
        
        trunk_config = trunk_config.copy()
        trunk_output_size = n_basis_functions
        trunk_config['layers'].append(trunk_output_size)
        trunk_networks = torch.nn.ModuleList()

        for t in range(model.n_outputs):
            t = model.create_network(trunk_config)
            trunk_networks.append(t)

        self.trunk_output_dim = trunk_output_size

        branch_config = branch_config.copy()
        branch_output_size = n_basis_functions * model.n_outputs
        branch_config['layers'].append(branch_output_size)
        branch = model.create_network(branch_config)
        branch_networks = torch.nn.ModuleList([branch])

        self.branch_output_dim = branch_output_size

        return branch_networks, trunk_networks


    def forward(self, model, data_branch, data_trunk, matrices_branch=None, matrices_trunk=None):
        branch_out = (
            matrices_branch[0]
            if matrices_branch is not None
            else model.get_branch_output(0, data_branch)
        )
        outputs = []

        for i in range(model.n_outputs):
            trunk_out = matrices_trunk[i] if matrices_trunk is not None else model.get_trunk_output(i, data_trunk)
            N = trunk_out.shape[-1]
            branch_out_split = branch_out[i * N : (i + 1) * N , : ]
            output = torch.matmul(trunk_out, branch_out_split).T
            outputs.append(output)

        return tuple(outputs)