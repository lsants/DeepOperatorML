from __future__ import annotations
import torch
import logging
from .base import TrainingStrategy
from typing import TYPE_CHECKING
from ...utilities.metrics.errors import ERROR_METRICS
from ..optimization.optimizers.optimizer_factory import create_optimizer, create_scheduler

if TYPE_CHECKING:
    from ..deeponet import DeepONet
    from ..config import ModelConfig
    from .config import PODConfig

logger = logging.getLogger(name=__name__)

class PODStrategy(TrainingStrategy):
    def __init__(self, config: 'PODConfig'):
        super().__init__(config)
        self.error_metric = ERROR_METRICS[config.error.lower()]

    def prepare_components(self, model_config: 'ModelConfig'):
        if not isinstance(self.config, PODConfig):
            raise TypeError("PODStrategy requires PODConfig")
        model_config.branch.component_type = "branch_neural"
        model_config.trunk.component_type = "pod"
        model_config.trunk.basis = self.config.pod_basis

    def setup_training(self, model: 'DeepONet'):
        trainable_params = self._get_trainable_parameters(model)
        if not trainable_params:
            raise ValueError("No trainable parameters found in the model.")
        self.train_schedule = []
        for spec in self.config.optimizer_scheduler:
            optimizer = create_optimizer(spec, trainable_params)
            scheduler = create_scheduler(spec, optimizer)
            self.train_schedule.append((optimizer, scheduler, spec.epochs))

    def _get_trainable_parameters(self, model: 'DeepONet'):
        trainable_params = []
        for name, param in model.branch.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def get_train_schedule(self) -> list[tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]]:
        if not hasattr(self, 'train_schedule'):
            raise ValueError("Training schedule not set up. Call setup_training first.")
        return self.train_schedule

    def check_phase_transition(self, epoch: int) -> bool:
        return False

    def execute_phase_transition(self, model: 'DeepONet'):
        raise NotImplementedError("POD strategy has no phase transitions")

    def validation_enabled(self) -> bool:
        return True
    
    def strategy_specific_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
        relative_error = (self.error_metric(y_true - y_pred) / self.error_metric(y_true))
        return relative_error.item()

    def get_optimizer_scheduler(self):
        return self.config.optimizer_scheduler