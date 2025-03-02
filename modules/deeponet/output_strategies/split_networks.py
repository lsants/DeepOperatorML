import logging
import torch
from .output_handling_base import OutputHandlingStrategy
from ...utilities.log_functions import pprint_layer_dict

logger = logging.getLogger(__name__)


class SplitNetworksStrategy(OutputHandlingStrategy):
    def __init__(self):
        super().__init__()

    def get_basis_config(self):
        return {'type': 'multiple'}

    def configure_networks(self, model, branch_config, trunk_config, **kwargs):
        pod_basis = getattr(model, 'pod_basis', None)
        n_basis_functions = model.n_basis_functions

        if pod_basis is not None:
            if pod_basis.shape[0] < 2:
                raise ValueError(
                    "Split networks strategy expects multiple sets of basis functions with shape (n_basis_sets, n_features, n_modes), with n_basis_sets > 1.")
            n_basis_functions = pod_basis.shape[1]
            model.n_basis_functions = n_basis_functions

        self.trunk_output_size = n_basis_functions * model.n_outputs
        self.n_trunk_outputs = self.trunk_output_size // model.n_basis_functions
        trunk_config = trunk_config.copy()
        trunk_config['layers'].append(self.trunk_output_size)
        
        trunk_network = model.create_network(trunk_config)

        self.branch_output_size = self.trunk_output_size
        self.n_branch_outputs = self.n_trunk_outputs
        branch_config = branch_config.copy()
        branch_config['layers'].append(self.branch_output_size)
        
        branch_network = model.create_network(branch_config)

        logger.info(f"\nNumber of Branch outputs: {self.n_branch_outputs}\n")
        logger.info(f"\nNumber of Trunk outputs: {self.n_trunk_outputs}\n")
        logger.info(
            f"\nBranch layer sizes: {pprint_layer_dict(branch_config['layers'])}\n")
        logger.info(
            f"\nTrunk layer sizes: {pprint_layer_dict(trunk_config['layers'])}\n")

        logger.info(
            f"\nBranch network size: {self.branch_output_size}\nTrunk network size: {self.trunk_output_size}\n")

        return branch_network, trunk_network

    def forward(self, model, data_branch, data_trunk, matrix_branch=None, matrix_trunk=None):
        branch_out = (
            matrix_branch
            if matrix_branch is not None
            else model.get_branch_output(data_branch)
        )
        trunk_out = (
            matrix_trunk
            if matrix_trunk is not None
            else model.get_trunk_output(data_trunk)
        )
        outputs = []

        for i in range(model.n_outputs):
            N = model.n_basis_functions
            trunk_out_split = trunk_out[ :, i * N: (i + 1) * N]
            branch_out_split = branch_out[i * N: (i + 1) * N, : ]
            output = torch.matmul(trunk_out_split, branch_out_split).T
            outputs.append(output)

        return tuple(outputs)
