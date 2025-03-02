import logging
import torch
from .output_handling_base import OutputHandlingStrategy
from ...utilities.log_functions import pprint_layer_dict

logger = logging.getLogger(__name__)


class ShareBranchStrategy(OutputHandlingStrategy):
    """Use a single set of basis functions and one single input function mapping
       to learn all outputs.
    """

    def __init__(self):
        super().__init__()

    def get_basis_config(self):
        """
        Specifies that this strategy requires a single set of basis functions for both real and imaginary parts.

        Returns:
            dict: Basis configuration.
        """
        return {'type': 'multiple'}

    def configure_networks(self, model, branch_config, trunk_config, **kwargs):
        """
        Configures the networks for the ShareBranchStrategy.

        Args:
            model (DeepONet): The model instance.
            branch_config (dict): Initial branch configuration.
            trunk_config (dict): Initial trunk configuration.
            pod_basis (torch.Tensor): Tensor with POD Basis computed from data. Optional (used only in POD).

        Returns:
            tuple: (branch_network, trunk_network) where both are ModuleLists.
        """
        pod_basis = getattr(model, 'pod_basis', None)
        n_basis_functions = model.n_basis_functions

        if pod_basis is not None:
            n_basis_functions = pod_basis.shape[-1] // model.n_outputs
            model.n_basis_functions = n_basis_functions

        self.trunk_output_size = n_basis_functions * model.n_outputs
        self.n_trunk_outputs = self.trunk_output_size // model.n_basis_functions
        trunk_config = trunk_config.copy()
        trunk_config['layers'].append(self.trunk_output_size)

        trunk_network = model.create_network(trunk_config)

        self.branch_output_size = n_basis_functions
        self.n_branch_outputs = self.branch_output_size // model.n_basis_functions
        branch_config = branch_config.copy()
        branch_config['layers'].append(self.branch_output_size)

        branch_network = model.create_network(branch_config)

        logger.debug(f"\nNumber of Branch outputs: {self.n_branch_outputs}\n")
        logger.debug(f"\nNumber of Trunk outputs: {self.n_trunk_outputs}\n")
        logger.debug(
            f"\nBranch layer sizes: {pprint_layer_dict(branch_config['layers'])}\n")
        logger.debug(
            f"\nTrunk layer sizes: {pprint_layer_dict(trunk_config['layers'])}\n")

        logger.debug(
            f"\nBranch network size: {self.branch_output_size}\nTrunk network size: {self.trunk_output_size}\n")

        return branch_network, trunk_network

    def forward(self, model, data_branch, data_trunk, matrix_branch=None, matrix_trunk=None):
        N = model.n_basis_functions

        branch_out = (
            matrix_branch
            if matrix_branch is not None
            else model.get_branch_output(data_branch)
        )
        if data_trunk is None and matrix_trunk is None:
            identity = torch.eye(N).to(dtype=branch_out.dtype, 
                                   device=branch_out.device)
            I_generator = (identity for _ in range(model.n_outputs))
            matrix_trunk = torch.cat(tuple(I_generator), dim=1)
        trunk_out = (
            matrix_trunk
            if matrix_trunk is not None
            else model.get_trunk_output(data_trunk)
        )
        outputs = []
        
        for i in range(model.n_outputs):
            if trunk_out is not None:
                trunk_out_split = trunk_out[ : , i * N: (i + 1) * N]
            else:
                trunk_out = identity
            output = torch.matmul(trunk_out_split, branch_out).T
            outputs.append(output)

        return tuple(outputs)
