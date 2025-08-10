import logging
import torch
from typing import TYPE_CHECKING
from src.modules.models.deeponet.components.output_handler.config import OutputConfig
from src.modules.models.deeponet.components.output_handler.registry import OutputRegistry
from src.modules.models.deeponet.components.output_handler.protocol import OutputHandler
if TYPE_CHECKING:
    from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig

logger = logging.getLogger(__name__)


@OutputRegistry.register("shared_branch")
class SharedBranchHandler(OutputHandler):
    def __init__(self, config: OutputConfig):
        self.num_channels = config.num_channels
        self.dims_adjust = config.dims_adjust

    def adjust_dimensions(self, config: 'DeepONetConfig'):
        """Only modifies trunk output dim"""
        if not self.dims_adjust:
            return

        if config.trunk.inner_config is not None:
            config.trunk.inner_config.output_dim *= self.num_channels  # type: ignore


        if config.branch.output_dim is None:
            raise ValueError(
                "Branch output dimension must be set for shared branch handler."
            )
        original_output_dims = config.trunk.output_dim
        if original_output_dims is None:
            raise ValueError(
                "Trunk output dimension must be set for shared branch handler."
            )
        config.branch.output_dim = original_output_dims
        config.trunk.output_dim = original_output_dims * self.num_channels

    def combine(self, branch_out: torch.Tensor, trunk_out: torch.Tensor) -> torch.Tensor:
        # trunk_out: (T, C*P) -> (T, C, P)
        # branch_out: (B, P)
        T = trunk_out.size(0)
        trunk_reshaped = trunk_out.view(
            T, self.num_channels, -1)  # (T, C, P)
        return torch.einsum('bp,tcp->btc', branch_out, trunk_reshaped)
