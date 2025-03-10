import logging
import torch
from .output_handling_base import OutputHandling
from ...utilities.log_functions import pprint_layer_dict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class SplitOutputsHandling(OutputHandling):
    def __init__(self):
        super().__init__()

    @property
    def BASIS_CONFIG(self):
        return 'multiple'

    def configure_components(self, model, branch_config: dict, trunk_config: dict, **kwargs) -> tuple:
        processed_trunk_config = self.config_basis(model, trunk_config)

        n_basis_functions = model.n_basis_functions
        trunk_output_size = n_basis_functions * model.n_outputs
        branch_output_size = n_basis_functions * model.n_outputs


        branch, trunk = self.create_components(model, branch_config, processed_trunk_config, branch_output_size, trunk_output_size)

        logger.debug(f"SplitOutputHandling: Computed trunk output size: {trunk_output_size}")
        logger.debug(f"SplitOutputHandling: Computed branch output size: {branch_output_size}")
        logger.debug(f"SplitOutputHandling: Trunk layers: {pprint_layer_dict(processed_trunk_config.get('layers', []))}")
        logger.debug(f"SplitOutputHandling: Branch layers: {pprint_layer_dict(branch_config.get('layers', []))}")

        return branch, trunk

    def forward(self, model: 'DeepONet', branch_out: torch.Tensor, trunk_out: torch.Tensor) -> tuple[torch.Tensor]: 
        """
        For SplitOutputsHandling, both trunk and branch outputs are split into segments,
        and each output is computed by multiplying the corresponding trunk and branch slices.
        """
        N = model.n_basis_functions
        outputs = []
        for i in range(model.n_outputs):
            trunk_slice = trunk_out[ : , i * N : (i + 1) * N]
            branch_slice = branch_out[ : , i * N : (i + 1) * N]
            output = torch.matmul(trunk_slice, branch_slice.T).T
            outputs.append(output)
        return tuple(outputs)

