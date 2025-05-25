import logging
import torch
from typing import TYPE_CHECKING
from .config import OutputConfig
from .registry import OutputRegistry
from .protocol import OutputHandler
from .....exceptions import ConfigValidationError
if TYPE_CHECKING:
    from ...config import ModelConfig

logger = logging.getLogger(__name__)

@OutputRegistry.register("shared_trunk")
class SharedTrunkHandler(OutputHandler):
    def __init__(self, config: OutputConfig):
        if config.num_channels < 1:
            raise ConfigValidationError(
                "Channel count must be ≥1 for shared trunk",
                section="output"
            )
        self.num_channels = config.num_channels
        self.basis_adjust = config.basis_adjust

    def adjust_dimensions(self, config: 'ModelConfig'):
        """Only modifies branch output dim"""
        if not self.basis_adjust: return
        
        original_basis = config.trunk.output_dim
        config.branch.output_dim = original_basis * self.num_channels

    def combine(self, branch_out: torch.Tensor, trunk_out: torch.Tensor) -> torch.Tensor:
        batch_size = branch_out.size(0)
        basis = trunk_out.size(-1)
        
        branch_reshaped = branch_out.view(batch_size, self.num_channels, basis)
        return torch.einsum('bci,bi->bc', branch_reshaped, trunk_out)
