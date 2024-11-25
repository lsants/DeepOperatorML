# modules/model/output_strategies/single_trunk_single_branch.py

import torch
from .output_handling_base import OutputHandlingStrategy

class SingleTrunkSplitBranchStrategy(OutputHandlingStrategy):
    """Use a single set of basis functions and one single input function mapping
       to learn all outputs.
    """
    def __init__(self):
        self.branch_output_dim = None
        self.trunk_output_dim = None
    def configure_networks(self, model, branch_config, trunk_config, **kwargs):
        """
        Configures the networks for the SingleTrunkSplitBranchStrategy.

        Args:
            model (DeepONet): The model instance.
            branch_config (dict): Initial branch configuration.
            trunk_config (dict): Initial trunk configuration.
            basis_functions (int): Number of basis functions to use.

        Returns:
            tuple: (branch_networks, trunk_networks) where both are ModuleLists.
        """
        basis_functions = kwargs.get('basis_functions')
        trunk_config = trunk_config.copy()
        trunk_config['layers'][-1] = basis_functions

        trunk = model.create_network(trunk_config)
        trunk_networks = torch.nn.ModuleList([trunk])

        branch_config = branch_config.copy()
        branch_output_size = basis_functions * model.n_outputs
        branch_config['layers'][-1] = branch_output_size

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

