import torch
from typing import Literal, Protocol, TYPE_CHECKING
if TYPE_CHECKING:
    from ...config import ModelConfig


class OutputHandler(Protocol):
    num_channels: int
    def adjust_dimensions(self, config: 'ModelConfig'): ...

    def combine(self, branch_out: torch.Tensor,
                trunk_out: torch.Tensor) -> torch.Tensor: ...
