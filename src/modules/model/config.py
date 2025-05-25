from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from .components.branch.config import BranchConfig, BranchConfig
from .components.trunk.config import TrunkConfig, TrunkConfig
from .components.output_handler.config import OutputConfig
from .components.rescaling.config import RescalingConfig
if TYPE_CHECKING:
    from .training_strategies.base import StrategyConfig
@dataclass
class ModelConfig:
    branch: BranchConfig
    trunk: TrunkConfig
    output: OutputConfig
    rescaling: RescalingConfig
    strategy: StrategyConfig