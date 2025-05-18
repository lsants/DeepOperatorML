import logging
import torch
from .output_handling_base import OutputHandling
from ...utilities.log_functions import pprint_layer_dict
from typing import TYPE_CHECKING, Any, Literal
if TYPE_CHECKING:
    from modules.model.deeponet import DeepONet
    from modules.model.components import BaseBranch, BaseTrunk

logger = logging.getLogger(__name__)

class SplitOutputsHandling(OutputHandling):
    def __init__(self):
        super().__init__()

    @property
    def BASIS_CONFIG(self) -> Literal['multiple']:
        return 'multiple'

    def configure_components(self, model: 'DeepONet', branch_config: dict[str, Any], trunk_config: dict[str, Any]) -> tuple['BaseBranch', 'BaseTrunk']:
        processed_trunk_config = self.config_basis(model=model, trunk_config=trunk_config)

        n_basis_functions = model.n_basis_functions
        self.trunk_output_size = n_basis_functions * model.n_outputs
        self.branch_output_size = n_basis_functions * model.n_outputs


        branch, trunk = self.create_components(model=model, 
                                               branch_config=branch_config, 
                                               trunk_config=processed_trunk_config, 
                                               branch_output_size=self.branch_output_size,
                                               trunk_output_size=self.trunk_output_size)

        logger.debug(f"SplitOutputHandling: Computed trunk output size: {self.trunk_output_size}")
        logger.debug(f"SplitOutputHandling: Computed branch output size: {self.branch_output_size}")
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

