from dataclasses import dataclass
from typing import Literal


@dataclass
class OutputConfig:
    handler_type: Literal["split", "shared", "single"]
    num_channels: int = 1
    basis_adjust: bool = True  # Whether to modify trunk layers

@dataclass
class RescalingConfig:
    exponent: float
    basis_functions: int