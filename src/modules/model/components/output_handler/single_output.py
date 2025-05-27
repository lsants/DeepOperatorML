from __future__ import annotations
import torch
import logging
from .protocol import OutputHandler
from .config import OutputConfig
from .....exceptions import ConfigValidationError
from ..output_handler.registry import OutputRegistry
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...config import ModelConfig


logger = logging.getLogger(__name__)

@OutputRegistry.register("single_output")
class SingleOutputHandler(OutputHandler):
    def __init__(self, config: OutputConfig):
        if config.num_channels != 1:
            raise ConfigValidationError(
                "Single output requires exactly 1 channel",
                section="output"
            )

    def adjust_dimensions(self, config: 'ModelConfig'):
        """No dimension adjustments"""
        pass

    def combine(self, branch_out: torch.Tensor, trunk_out: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ai,bi->ab', branch_out, trunk_out)
