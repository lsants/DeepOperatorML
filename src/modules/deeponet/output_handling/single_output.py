import logging
import torch
from .output_handling_base import OutputHandling
from ...utilities.log_functions import pprint_layer_dict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class SingleOutputHandling(OutputHandling):
    """Use a single set of basis functions and one single input function mapping
       to learn one single output.
    """
    def __init__(self):
        super().__init__()
        
    @property
    def BASIS_CONFIG(self):
        return 'single'
    
    def configure_components(self, model: 'DeepONet', branch_config: dict, trunk_config: dict, **kwargs) -> tuple:
        processed_trunk_config = self.config_basis(model, trunk_config)

        n_basis_functions = model.n_basis_functions
        trunk_output_size = n_basis_functions
        branch_output_size = n_basis_functions

        branch, trunk = self.create_components(model, branch_config, processed_trunk_config, branch_output_size, trunk_output_size)

        logger.debug(f"SingleOutputHandling: Computed trunk output size: {trunk_output_size}")
        logger.debug(f"SingleOutputHandling: Computed branch output size: {branch_output_size}")
        logger.debug(f"SingleOutputHandling: Trunk layers: {pprint_layer_dict(processed_trunk_config.get('layers', []))}")
        logger.debug(f"SingleOutputHandling: Branch layers: {pprint_layer_dict(branch_config.get('layers', []))}")

        return branch, trunk
    
    def forward(self, model: 'DeepONet', branch_out: torch.Tensor, trunk_out: torch.Tensor) -> tuple[torch.Tensor]: 
        """
        For SingleOutputHandling, a single aggregated output is produced by directly multiplying
        the trunk and branch outputs.
        """
        output = torch.matmul(trunk_out, branch_out.T).T
        return (output,)


