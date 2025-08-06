from __future__ import annotations
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.modules.models.deeponet.components.output_handler.protocol import OutputHandler
    from src.modules.models.deeponet.components.rescaling.rescaler import Rescaler

class DeepONet(torch.nn.Module):
    def __init__(self, branch: torch.nn.Module, trunk: torch.nn.Module, bias: torch.nn.Module, output_handler: OutputHandler, rescaler: Rescaler):
        """DeepONet model (Lu, et al. 2019), universal approximation theorem-based architecture composed of a branch and a trunk net.

        Args:
            branch (torch.nn.Module): Branch network, used to learn an operator's input function.
            trunk (torch.nn.Module): Trunk network, used to learn the operator's basis mapping.
            output_handler (OutputHandler): Defines how the DeepONet's inner product will be conducted depending on the number of channels.
            rescaling (Rescaling): Determines how the DeepONet's output is rescaled in function of the number of basis functions.
        """
        super().__init__()
        self.branch: torch.nn.Module = branch
        self.trunk: torch.nn.Module = trunk
        self.bias: torch.nn.Module =  bias
        self.output_handler: OutputHandler = output_handler
        self.rescaler: Rescaler = rescaler

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:

        branch_out = self.branch(branch_input)
        trunk_out = self.trunk(trunk_input)

        dot_product = self.output_handler.combine(branch_out, trunk_out)
        output = self.bias(dot_product)

        return self.rescaler(output)

