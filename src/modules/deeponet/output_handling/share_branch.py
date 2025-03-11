import logging
import torch
from .output_handling_base import OutputHandling
from ..training_strategies import TwoStepTrainingStrategy, PODTrainingStrategy
from ....exceptions import InvalidStrategyCombinationError
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
        if isinstance(model.training_strategy, TwoStepTrainingStrategy) or isinstance(model.training_strategy, PODTrainingStrategy):
            raise InvalidStrategyCombinationError(
                strategy=model.training_strategy,
                handling='ShareBranchHandling',
                reason=(
                    "The fixed trunk restricts the optimization problem "
                    "(a single branch representation is insufficient to fit the solution space)."
                ),
                suggestion="Please choose a different output handling."
            )
            
        processed_trunk_config = self.config_basis(model, trunk_config)

        n_basis_functions = model.n_basis_functions
        self.trunk_output_size = n_basis_functions * model.n_outputs
        self.branch_output_size = n_basis_functions

        branch, trunk = self.create_components(model, branch_config, processed_trunk_config, self.branch_output_size, self.trunk_output_size)

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