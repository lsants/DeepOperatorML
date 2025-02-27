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
        self.branch_output_size = None
        self.trunk_output_size = None
        self.num_trunks = None
        self.num_branches = None

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
            tuple: (branch_networks, trunk_networks) where both are ModuleLists.
        """
        pod_basis = getattr(model, 'pod_basis', None)
        n_basis_functions = model.n_basis_functions

        if pod_basis is not None:
            if pod_basis.shape[0] < 2 and model.n_outputs > 1:
                raise ValueError(f"ShareBranchStrategy expects a set of multiple basis functions for each output with shape (n_outputs, n_features, n_modes).\n \
                                 The basis set is of shape {pod_basis.shape}")
            n_basis_functions = pod_basis.shape[-1]
            model.n_basis_functions = n_basis_functions

        trunk_config = trunk_config.copy()
        trunk_output_size = n_basis_functions * model.n_outputs
        trunk_config['layers'].append(trunk_output_size)

        trunk = model.create_network(trunk_config)
        trunk_networks = torch.nn.ModuleList([trunk])

        self.trunk_output_size = trunk_output_size
        self.num_trunks = len(trunk_networks)

        branch_config = branch_config.copy()
        branch_output_size = n_basis_functions
        branch_config['layers'].append(branch_output_size)

        branch = model.create_network(branch_config)
        branch_networks = torch.nn.ModuleList([branch])

        self.branch_output_size = branch_output_size
        self.num_branches = len(branch_networks)

        logger.info(f"\nNumber of Branch networks: {self.num_branches}\n")
        logger.info(f"\nNumber of Trunk networks: {self.num_trunks}\n")
        logger.info(
            f"\nBranch layer sizes: {pprint_layer_dict(branch_config['layers'])}\n")
        logger.info(
            f"\nTrunk layer sizes: {pprint_layer_dict(trunk_config['layers'])}\n")

        logger.info(
            f"\nBranch network size: {branch_output_size}\nTrunk network size: {trunk_output_size}\n")

        return branch_networks, trunk_networks

    def forward(self, model, data_branch, data_trunk, matrices_branch=None, matrices_trunk=None):
        N = model.n_basis_functions

        branch_out = (
            matrices_branch[0]
            if matrices_branch is not None
            else model.get_branch_output(0, data_branch)
        )

        if data_trunk is None and matrices_trunk is None:
            identity = torch.eye(N).to(dtype=branch_out.dtype, 
                                   device=branch_out.device)
            I_generator = (identity for _ in range(model.n_outputs))
            matrices_trunk = [torch.cat(tuple(I_generator), dim=1)]

        print("THIS SHOULD RUN 3", matrices_trunk[0].shape if matrices_trunk is not None else data_trunk)
        trunk_out = (
            matrices_trunk[0]
            if matrices_trunk is not None
            else model.get_trunk_output(0, data_trunk)
        )

        print("THIS SHOULD RUN 4", trunk_out.shape)

        print("THIS SHOULD RUN 5", trunk_out.shape)



        outputs = []
        
        for i in range(model.n_outputs):
            if trunk_out is not None:
                trunk_out_split = trunk_out[ : , i * N: (i + 1) * N]
            else:
                trunk_out = identity
            output = torch.matmul(trunk_out_split, branch_out).T
            outputs.append(output)
        print("THIS SHOULD RUN 6", trunk_out.shape)

        return tuple(outputs)
