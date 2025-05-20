import logging
import torch
from .base import OutputHandling
from ...utilities.log_functions import pprint_layer_dict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.model.deeponet import DeepONet
    from modules.model.components import BaseBranch, BaseTrunk


logger = logging.getLogger(__name__)

class ShareTrunkHandling(OutputHandling):
    """Use a single set of basis functions and one single input function mapping
       to learn all output.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def BASIS_CONFIG(self) -> str:
        return 'single'

    def configure_components(self, model: 'DeepONet', branch_config: dict, trunk_config: dict) -> tuple:
        processed_trunk_config = self.config_basis(model, trunk_config)

        n_basis_functions = model.n_basis_functions
        self.trunk_output_size = n_basis_functions
        self.branch_output_size = n_basis_functions * model.n_outputs

        branch, trunk = self.create_components(model, branch_config, processed_trunk_config, self.branch_output_size, self.trunk_output_size)

        logger.debug(f"ShareTrunkHandling: Computed trunk output size: {self.trunk_output_size}")
        logger.debug(f"ShareTrunkHandling: Computed branch output size: {self.branch_output_size}")
        logger.debug(f"ShareTrunkHandling: Trunk layers: {pprint_layer_dict(processed_trunk_config.get('layers', []))}")
        logger.debug(f"ShareTrunkHandling: Branch layers: {pprint_layer_dict(branch_config.get('layers', []))}")

        return branch, trunk
    
    def forward(self, model: 'DeepONet', branch_out: torch.Tensor, trunk_out: torch.Tensor) -> tuple[torch.Tensor]: 
        """
        For ShareTrunkHandling, we assume that the trunk output is a single fixed basis.
        The branch output is split into slices (one per operator output).
        Each output is computed by multiplying the trunk output with the corresponding slice of branch output.
        """
        N = model.n_basis_functions
        outputs = []
        for i in range(model.n_outputs):
            branch_slice = branch_out[ : , i * N : (i + 1) * N]
            output = torch.matmul(trunk_out, branch_slice.T).T
            outputs.append(output)
        return tuple(outputs)