# modules/model/deeponet.py
from __future__ import annotations
import torch
import logging
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .output_handling.output_handling_base import OutputHandling
    from .training_strategies.training_strategy_base import TrainingStrategy

logger = logging.getLogger(__name__)

class DeepONet(torch.nn.Module):
    def __init__(self, branch_config: dict[str, Any], trunk_config: dict[str, Any], output_handling: 'OutputHandling', training_strategy: 'TrainingStrategy', n_outputs: int, n_basis_functions: int, **kwargs:Any) -> None:
        """Initializes the DeepONet model with specified strategies.

        Args:
            branch_config (dict): Configuration of the branch networks.
            trunk_config (dict): Configuration of the trunk networks.
            output_handling (OutputHandling): Strategy for handling the outputs.
            training_strategy (TrainingStrategy): Strategy for training.
            n_outputs (int): Number of outputs.
            n_basis_functions (int): Number of basis functions.

        """
        super(DeepONet, self).__init__()
        self.n_outputs: int = n_outputs
        self.n_basis_functions: int = n_basis_functions
        self.output_handling: 'OutputHandling' = output_handling
        self.training_strategy: 'TrainingStrategy' = training_strategy

        trunk_config = self.training_strategy.get_trunk_config(
            trunk_config=trunk_config)
        branch_config = self.training_strategy.get_branch_config(
            branch_config=branch_config)

        self.branch, self.trunk = self.output_handling.configure_components(
            model=self, branch_component=branch_config, trunk_component=trunk_config, **kwargs
        )

        self.training_strategy.prepare_for_training(model=self, **kwargs)

    def forward(self, branch_input: torch.Tensor | None = None, trunk_input: torch.Tensor | None = None) -> tuple[torch.Tensor]:
        return self.training_strategy.forward(model=self, branch_input=branch_input, trunk_input=trunk_input)
