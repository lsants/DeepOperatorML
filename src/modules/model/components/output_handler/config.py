from typing import Literal
from dataclasses import dataclass

@dataclass
class OutputConfig:
    handler_type: Literal["split_outputs", "shared_trunk", "single_output"]
    num_channels: int = 1
    basis_adjust: bool = True