import logging
import torch
from .output_handling_base import OutputHandling
from ...utilities.log_functions import pprint_layer_dict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class ShareBranchHandling(OutputHandling):
    def __init__(self):
        super().__init__()

    @property
    def BASIS_CONFIG(self):
        return 'multiple'

    def configure_components(self, model, branch_config: dict, trunk_config: dict, **kwargs) -> tuple:
        n_basis_functions = model.n_basis_functions
        trunk_output_size = n_basis_functions * model.n_outputs
        branch_output_size = n_basis_functions
       
        branch, trunk = self.create_components(model, branch_config, trunk_config, branch_output_size, trunk_output_size)

        logger.debug(f"ShareBranchHandling: Computed trunk output size: {trunk_output_size}")
        logger.debug(f"ShareBranchHandling: Computed branch output size: {branch_output_size}")
        logger.debug(f"ShareBranchHandling: Trunk layers: {pprint_layer_dict(trunk_config.get('layers', []))}")
        logger.debug(f"ShareBranchHandling: Branch layers: {pprint_layer_dict(branch_config.get('layers', []))}")

        return branch, trunk
    
    def forward(self, model: 'DeepONet', branch_out: torch.Tensor, trunk_out: torch.Tensor) -> tuple[torch.Tensor]: 
        """
        For ShareBranchHandling, the branch output remains un-split, while the trunk output
        is split into segments of size equal to model.n_basis_functions (one segment per output).
        The final output for each operator is computed by multiplying the corresponding trunk slice 
        with the branch output.
        """
        N = model.n_basis_functions
        outputs = []
        for i in range(model.n_outputs):
            trunk_slice = trunk_out[ : , i * N : (i + 1) * N]
            output = torch.matmul(trunk_slice, branch_out.T).T
            outputs.append(output)
        return tuple(outputs)