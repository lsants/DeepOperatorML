import logging
import torch
from .output_handling_base import OutputHandlingStrategy
from ...utilities.log_functions import pprint_layer_dict

logger = logging.getLogger(__name__)


class SplitNetworksStrategy(OutputHandlingStrategy):
    def __init__(self):
        self.branch_output_size = None
        self.trunk_output_size = None
        self.num_trunks = None
        self.num_branches = None

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

        trunk_config = trunk_config.copy()
        trunk_output_size = n_basis_functions * model.n_outputs
        trunk_config['layers'].append(trunk_output_size)

        trunk = model.create_network(trunk_config)
        trunk_networks = torch.nn.ModuleList([trunk])

        self.num_trunks = len(trunk_networks)
        self.trunk_output_size = trunk_output_size

        branch_config = branch_config.copy()
        branch_output_size = n_basis_functions * model.n_outputs
        branch_config['layers'].append(branch_output_size)

        branch = model.create_network(branch_config)
        branch_networks = torch.nn.ModuleList([branch])

        self.num_branches = len(branch_networks)
        self.branch_output_size = branch_output_size

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
        branch_out = (
            matrices_branch[0]
            if matrices_branch is not None
            else model.get_branch_output(0, data_branch)
        )
        trunk_out = (
            matrices_trunk[0]
            if matrices_trunk is not None
            else model.get_trunk_output(0, data_trunk)
        )
        outputs = []

        for i in range(model.n_outputs):
            N = model.n_basis_functions
            trunk_out_split = trunk_out[:, i * N: (i + 1) * N]
            branch_out_split = branch_out[i * N: (i + 1) * N, :]
            output = torch.matmul(trunk_out_split, branch_out_split).T
            outputs.append(output)

        return tuple(outputs)
