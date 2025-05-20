from typing import Union
from dataclasses import dataclass

@dataclass
class SingleOutputConfig:
    pass  # No additional params needed

@dataclass
class SplitOutputConfig:
    num_outputs: int  # Must match len(targets)

@dataclass
class SharedTrunkConfig:
    shared_dim: int  # Dimension of shared trunk features

OutputHandlingConfig = Union[
    SingleOutputConfig,
    SplitOutputConfig,
    SharedTrunkConfig
]