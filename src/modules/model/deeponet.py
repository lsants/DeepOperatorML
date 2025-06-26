# modules/model/deeponet.py
from __future__ import annotations
import torch
import logging
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .components.output_handler.protocol import OutputHandler
    from .components.rescaling.rescaler import Rescaler

logger = logging.getLogger(__name__)


class DeepONet(torch.nn.Module):
    def __init__(self, branch: torch.nn.Module,
                 trunk: torch.nn.Module,
                 output_handler: OutputHandler,
                 rescaler: Rescaler):
        """DeepONet model (Lu, et al. 2019), universal approximation theorem-based architecture composed of a branch and a trunk net.

        Args:
            branch (torch.nn.Module): Branch network, used to learn an operator's input function.
            trunk (torch.nn.Module): Trunk network, used to learn the operator's basis mapping.
            output_handler (OutputHandler): Defines how the DeepONet's inner product will be conducted depending on the number of channels.
            rescaling (Rescaling): Determines how the DeepONet's output is rescaled in function of the number of basis functions.
        """
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.output_handler = output_handler
        self.rescaler = rescaler
        self.bias = torch.nn.Parameter(
            torch.zeros(self.output_handler.num_channels))

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        branch_out = self.branch(branch_input)
        trunk_out = self.trunk(trunk_input)

        combined = self.output_handler.combine(
            branch_out, trunk_out)

        return self.rescaler(combined)
