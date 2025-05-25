from __future__ import annotations
import torch
import logging
from typing import Any
from .base import StrategyConfig
from .base import TrainingStrategy
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..deeponet import DeepONet
    from ..config import ModelConfig

logger = logging.getLogger(name=__name__)

class PODStrategy(TrainingStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def prepare_components(self, model_config: ModelConfig):
        if not isinstance(self.config, StrategyConfig):
            raise TypeError("PODStrategy requires StrategyConfig")
        model_config.trunk.component_type = "pod"
        model_config.trunk.basis = self.config.pod_basis

    def setup_training(self, model: DeepONet):
        model.trunk.requires_grad_(False)  # Freeze trunk

    def check_phase_transition(self, model: DeepONet, epoch: int) -> bool:
        return False

    def execute_phase_transition(self, model: DeepONet):
        raise NotImplementedError("POD strategy has no phase transitions")

