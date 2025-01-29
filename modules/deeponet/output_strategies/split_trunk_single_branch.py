# modules/model/output_strategies/split_trunk_single_branch.py

import torch
from .output_handling_base import OutputHandlingStrategy

class SplitTrunkSingleBranchStrategy(OutputHandlingStrategy):
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
        return {'type': 'multiple'}
    
    def configure_networks(self, model, branch_config, trunk_config, **kwargs):
        """
        Configures the networks for the SplitTrunkSingleBranchStrategy.

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
            if pod_basis.shape[0] < 2:
                raise ValueError("SplitTrunkSingleBranchStrategy expects a set of multiple basis functions for each output with shape (n_outputs, n_features, n_modes).")
            n_basis_functions = pod_basis.shape[-1]
            model.n_basis_functions = n_basis_functions

        trunk_config = trunk_config.copy()
        trunk_output_size = n_basis_functions * model.n_outputs
        trunk_config['layers'].append(trunk_output_size)

        trunk = model.create_network(trunk_config)
        trunk_networks = torch.nn.ModuleList([trunk])

        self.trunk_output_dim = trunk_output_size

        branch_config = branch_config.copy()
        branch_output_size = n_basis_functions
        branch_config['layers'].append(branch_output_size)

        branch = model.create_network(branch_config)
        branch_networks = torch.nn.ModuleList([branch])

        self.branch_output_dim = branch_output_size

        return branch_networks, trunk_networks
    
    def forward(self, model, data_branch, data_trunk, matrices_branch=None, matrices_trunk=None):
        mask_matrix = torch.cat(tuple(i for i in matrices_trunk), axis=1) if matrices_trunk is not None else None
        mask_data = torch.cat(tuple(model.get_trunk_output(i, data_trunk) for i in range(model.n_outputs)), axis=1)
        
        trunk_out = (
            mask_matrix
            if mask_matrix is not None
            else mask_data
        )

        branch_out = (
            matrices_branch[0]
            if matrices_branch is not None
            else model.get_branch_output(0, data_branch)
        )

        N = branch_out.shape[0] 

        outputs = []

        for i in range(model.n_outputs):
            trunk_out_split = trunk_out[ : , i * N : (i + 1) * N ]
            output = torch.matmul(trunk_out_split, branch_out).T
            outputs.append(output)

        return tuple(outputs)

