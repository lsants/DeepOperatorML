# modules/model/output_strategies/single_trunk_split_branch.py

import torch
from .output_handling_base import OutputHandlingStrategy

class SingleTrunkSplitBranchStrategy(OutputHandlingStrategy):
    """Use a single set of basis functions and one single input function mapping
       to learn all outputs.
    """
    def __init__(self):
        self.branch_output_dim = None
        self.trunk_output_dim = None
        
    def get_basis_config(self):
        """
        Specifies that this strategy requires a single set of basis functions for both real and imaginary parts.

        Returns:
            dict: Basis configuration.
        """
        return {'type': 'single'}
    
    def configure_networks(self, model, branch_config, trunk_config, **kwargs):
        """
        Configures the networks for the SingleTrunkSplitBranchStrategy.

        Args:
            model (DeepONet): The model instance.
            branch_config (dict): Initial branch configuration.
            trunk_config (dict): Initial trunk configuration.
            pod_basis (torch.Tensor): Tensor with POD Basis computed from data. Optional (used only in POD).

        Returns:
            tuple: (branch_networks, trunk_networks) where both are ModuleLists.
        """
        pod_basis = getattr(model, 'pod_basis', None)
        n_basis_functions = model.n_basis_functions

        if pod_basis is not None:
            if pod_basis.shape[0] != 1:
                raise ValueError("SingleTrunkSplitBranchStrategy expects a single set of basis functions with shape (1, n_features, n_modes).")
            single_basis = pod_basis[0]
            n_basis_functions = single_basis.shape[1]
            model.n_basis_functions = n_basis_functions

        trunk_config = trunk_config.copy()
        trunk_output_size = n_basis_functions
        trunk_config['layers'].append(trunk_output_size)

        trunk = model.create_network(trunk_config)
        trunk_networks = torch.nn.ModuleList([trunk])

        self.trunk_output_dim = trunk_output_size

        branch_config = branch_config.copy()
        branch_output_size = n_basis_functions * model.n_outputs
        branch_config['layers'].append(branch_output_size)

        branch = model.create_network(branch_config)
        branch_networks = torch.nn.ModuleList([branch])

        self.branch_output_dim = branch_output_size

        return branch_networks, trunk_networks
    
    def forward(self, model, data_branch, data_trunk, matrices_branch=None, matrices_trunk=None):
        trunk_out = (
            matrices_trunk[0]
            if matrices_trunk is not None
            else model.get_trunk_output(0, data_trunk)
        )

        branch_out = (
            matrices_branch[0]
            if matrices_branch is not None
            else model.get_branch_output(0, data_branch)
        )

        N = trunk_out.shape[-1] 

        outputs = []

        for i in range(model.n_outputs):
            branch_out_split = branch_out[ i * N : (i + 1) * N , : ]
            output = torch.matmul(trunk_out, branch_out_split).T
            outputs.append(output)

        return tuple(outputs)

