from __future__ import annotations
import torch
import logging
from .config import PODConfig
from .base import TrainingStrategy
from typing import TYPE_CHECKING
from ...utilities.metrics.errors import ERROR_METRICS
from ..optimization.optimizers.config import OptimizerSpec
from ..optimization.optimizers.optimizer_factory import create_optimizer, create_scheduler
from ..components.trunk import PODTrunk

if TYPE_CHECKING:
    from ..deeponet import DeepONet
    from ..config import ModelConfig

logger = logging.getLogger(name=__name__)


class PODStrategy(TrainingStrategy):
    def __init__(self, config: 'PODConfig'):
        super().__init__(config)
        self.error_metric = ERROR_METRICS[config.error.lower()]

    def prepare_components(self, model_config: 'ModelConfig'):
        if not isinstance(self.config, PODConfig):
            raise TypeError("PODStrategy requires PODConfig")
        model_config.branch.component_type = "neural_branch"
        model_config.trunk.component_type = "pod_trunk"
        model_config.trunk.pod_basis = self.config.pod_basis
        model_config.trunk.pod_mean = self.config.pod_mean

    def setup_training(self, model: 'DeepONet'):
        if not isinstance(model.trunk, PODTrunk):
            raise TypeError(
                "Model's trunk was not correctly defined as 'PODTrunk'.")
        if not isinstance(model.trunk.pod_mean, torch.Tensor):
            raise TypeError(
                "PODTrunk mean was not correctly setup.")

        model.trunk.requires_grad_(False)
        model.branch.requires_grad_(True)
        model.bias = torch.nn.Parameter(
            model.trunk.pod_mean, requires_grad=False)

        trainable_params = self._get_trainable_parameters(model)
        if not trainable_params:
            raise ValueError("No trainable parameters found in the model.")
        self.train_schedule = []
        for spec in self.config.optimizer_scheduler:  # type: ignore
            if isinstance(spec, dict):
                spec = OptimizerSpec(**spec)
            optimizer = create_optimizer(spec, trainable_params)
            scheduler = create_scheduler(spec, optimizer)
            self.train_schedule.append((spec.epochs, optimizer, scheduler))

    def _get_trainable_parameters(self, model: 'DeepONet'):
        trainable_params = []
        for name, param in model.branch.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def get_train_schedule(self) -> list[tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]]:
        if not hasattr(self, 'train_schedule'):
            raise ValueError(
                "Training schedule not set up. Call setup_training first.")
        return self.train_schedule

    def get_phases(self) -> list[str]:
        """Return phase names (e.g., ['phase1', 'phase2'])"""
        return ["POD"]

    def apply_gradient_constraints(self, model: DeepONet):
        """Optional gradient clipping/normalization"""
        pass

    def execute_phase_transition(self, model: 'DeepONet'):
        raise NotImplementedError("POD strategy has no phase transitions")

    def validation_enabled(self) -> bool:
        return True

    def strategy_specific_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
        relative_error = (self.error_metric(
            y_true - y_pred, dim=(0, 1)) / self.error_metric(y_true, dim=(0, 1)))
        strategy_metric = {
            **{f'error_{i[0]}': i[1].item() for i in enumerate(relative_error.detach())}
        }
        return strategy_metric

    def get_optimizer_scheduler(self):
        return self.config.optimizer_scheduler

    def should_transition_phase(self, current_phase: int, current_epoch: int) -> bool:
        return False
