from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import logging
from .protocol import OutputHandler
from .config import OutputConfig
from ..output_handler.registry import OutputRegistry
if TYPE_CHECKING:
    from ...config import ModelConfig


logger = logging.getLogger(__name__)

@OutputRegistry.register("split_outputs")
class SplitOutputsHandler(OutputHandler):
    def __init__(self, config: OutputConfig):
        self.num_channels = config.num_channels
        self.basis_adjust = config.basis_adjust

    def adjust_dimensions(self, config: 'ModelConfig'):
        if not self.basis_adjust:
            return
            
        config.trunk.output_dim *= self.num_channels
        config.branch.output_dim = config.trunk.output_dim

        # Verify branch output matches
        if config.branch.output_dim != config.trunk.output_dim:
            raise ValueError(f"Branch and trunk output sizes ({config.branch.output_dim}, {config.trunk.output_dim}) don't match.")

    def combine(self, branch_out: torch.Tensor, trunk_out: torch.Tensor) -> torch.Tensor:
        # Reshape to [batch, channels, basis]
        branch_reshaped = branch_out.view(-1, self.num_channels, trunk_out.size(-1))
        # Outer product per channel
        return torch.einsum('bci,bi->bc', branch_reshaped, trunk_out)

