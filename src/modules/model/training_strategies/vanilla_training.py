from __future__ import annotations
from dataclasses import dataclass
from .base import TrainingStrategy, StrategyConfig
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import ModelConfig
    from modules.model.deeponet import DeepONet

class VanillaStrategy(TrainingStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def prepare_components(self, model_config: ModelConfig):
        # Ensure neural architectures
        model_config.branch.component_type = "branch_neural"
        model_config.trunk.component_type = "trunk_neural"

    def setup_training(self, model: DeepONet):
        pass  # All components trainable

    def check_phase_transition(self, model: DeepONet, epoch: int) -> bool:
        return False  # No phase transitions

    def execute_phase_transition(self, model: DeepONet):
        raise NotImplementedError("Vanilla strategy has no phase transitions")
