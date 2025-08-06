from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import logging
from src.modules.models.deeponet.components.output_handler.protocol import OutputHandler
from src.modules.models.deeponet.components.output_handler.config import OutputConfig
from src.modules.models.deeponet.components.output_handler.registry import OutputRegistry
if TYPE_CHECKING:
    from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig

logger = logging.getLogger(__name__)


@OutputRegistry.register("split_outputs")
class SplitOutputsHandler(OutputHandler):
    def __init__(self, config: OutputConfig):
        self.num_channels = config.num_channels
        self.dims_adjust = config.dims_adjust

    def adjust_dimensions(self, config: 'DeepONetConfig'):
        if not self.dims_adjust:
            return

        config.trunk.output_dim *= self.num_channels  # type: ignore
        if config.trunk.inner_config is not None:
            config.trunk.inner_config.output_dim *= self.num_channels  # type: ignore

        config.branch.output_dim = config.trunk.output_dim

        if config.branch.output_dim != config.trunk.output_dim:
            raise ValueError(
                f"Branch and trunk output sizes ({config.branch.output_dim}, {config.trunk.output_dim}) don't match.")

    def combine(self, branch_out: torch.Tensor, trunk_out: torch.Tensor) -> torch.Tensor:
        # branch_out: (B, C*P) -> (B, C, P)
        # trunk_out: (T, C*P) -> (T, C, P)
        B = branch_out.size(0)
        T = trunk_out.size(0)
        branch_reshaped = branch_out.view(
            B, self.num_channels, -1)  # (B, C, P)
        trunk_reshaped = trunk_out.view(
            T, self.num_channels, -1)    # (T, C, P)

        result = torch.einsum(
            'bci,tci->btc', branch_reshaped, trunk_reshaped)  # (B, T, C)

        return result
