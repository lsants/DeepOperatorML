from __future__ import annotations
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal
from .components.branch.config import BranchConfig, BranchConfig
from .components.trunk.config import TrunkConfig, TrunkConfig
from .components.output_handler.config import OutputConfig
from .components.rescaling.config import RescalingConfig
from .training_strategies.config import StrategyConfig, VanillaConfig, TwoStepConfig, PODConfig
# if TYPE_CHECKING:
@dataclass
class ModelConfig:
    branch: BranchConfig
    trunk: TrunkConfig
    output: OutputConfig
    rescaling: RescalingConfig
    strategy: StrategyConfig

    def __post_init__(self):
        # Convert strategy dict to concrete subclass
        if isinstance(self.strategy, dict):
            strategy_data = self.strategy
            name = strategy_data["name"]
            
            # Map name to concrete class
            strategy_class = {
                "vanilla": VanillaConfig,
                "pod": PODConfig,
                "two_step": TwoStepConfig
            }[name]
            
            # Filter valid fields for the concrete class
            valid_fields = {f.name for f in fields(strategy_class)}
            filtered = {k: v for k, v in strategy_data.items() if k in valid_fields}
            

            # Replace with concrete instance
            self.strategy = strategy_class(**filtered)