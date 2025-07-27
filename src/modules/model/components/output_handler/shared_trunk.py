import logging
import torch
from typing import TYPE_CHECKING
from src.modules.model.components.output_handler.config import OutputConfig
from src.modules.model.components.output_handler.registry import OutputRegistry
from src.modules.model.components.output_handler.protocol import OutputHandler
if TYPE_CHECKING:
    from src.modules.model.config import ModelConfig

logger = logging.getLogger(__name__)


@OutputRegistry.register("shared_trunk")
class SharedTrunkHandler(OutputHandler):
    def __init__(self, config: OutputConfig):
        self.num_channels = config.num_channels
        self.dims_adjust = config.dims_adjust

    def adjust_dimensions(self, config: 'ModelConfig'):
        """Only modifies branch output dim"""
        if not self.dims_adjust:
            return

        if config.trunk.output_dim is None:
            raise ValueError(
                "Trunk output dimension must be set for shared trunk handler."
            )
        original_output_dims = config.trunk.output_dim
        config.branch.output_dim = original_output_dims * self.num_channels

    def combine(self, branch_out: torch.Tensor, trunk_out: torch.Tensor) -> torch.Tensor:
        # branch_out: (B, C*P) -> (B, C, P)
        # trunk_out: (T, P)
        B = branch_out.size(0)
        branch_reshaped = branch_out.view(
            B, self.num_channels, -1)  # (B, C, P)
        return torch.einsum('bcp,tp->btc', branch_reshaped, trunk_out)
