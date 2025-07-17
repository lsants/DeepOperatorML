from __future__ import annotations
import torch
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Type, Optional
from .components.branch.config import BranchConfig, BranchConfig
from .components.trunk.config import TrunkConfig, TrunkConfig
from .components.output_handler.config import OutputConfig
from .components.rescaling.config import RescalingConfig
from .components.bias.config import BiasConfig
from .training_strategies.config import StrategyConfig, VanillaConfig, TwoStepConfig, PODConfig
# if TYPE_CHECKING:


@dataclass
class ModelConfig:
    branch: BranchConfig
    trunk: TrunkConfig
    bias: BiasConfig
    output: OutputConfig
    rescaling: RescalingConfig
    strategy: StrategyConfig

    def __post_init__(self):
        if self.strategy is not None:
            self.strategy = self._convert_strategy(self.strategy)

    @classmethod
    def _convert_strategy(cls, strategy_data: Any):
        if isinstance(strategy_data, dict):
            name = strategy_data["name"]

            strategy_class: Type[StrategyConfig] = {
                "vanilla": VanillaConfig,
                "pod": PODConfig,
                "two_step": TwoStepConfig
            }[name]

            valid_fields = {f.name for f in fields(strategy_class)}

            filtered = {k: v for k, v in strategy_data.items()
                        if k in valid_fields}
            return strategy_class(**filtered)
        return strategy_data

    @classmethod
    def from_dict(cls, data: dict):

        if 'dtype' in data and isinstance(data['dtype'], str):
            data['dtype'] = getattr(torch, data['dtype'].split('.')[-1])

        converted = {}
        for field in fields(cls):
            if field.name not in data:
                continue

            value = data[field.name]

            if is_dataclass(field.type) and isinstance(value, dict):
                converted[field.name] = field.type.from_dict(value)
            elif (field.type is Optional[StrategyConfig] and isinstance(value, dict)):
                converted[field.name] = cls._convert_strategy(value)
            else:
                converted[field.name] = value

        return cls(**converted)
