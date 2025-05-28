from __future__ import annotations
import torch
from ..optimization.optimizers.config import OptimizerSpec
from .base import TrainingStrategy
from typing import TYPE_CHECKING
from ...utilities.metrics.errors import ERROR_METRICS
from ..optimization.optimizers.config import OptimizerSpec
from ..optimization.optimizers.optimizer_factory import create_optimizer, create_scheduler
if TYPE_CHECKING:
    from ..config import ModelConfig
    from modules.model.deeponet import DeepONet
    from .config import VanillaConfig

class VanillaStrategy(TrainingStrategy):
    def __init__(self, config: VanillaConfig):
        super().__init__(config)
        self.error_metric = ERROR_METRICS[config.error.lower()]

    def prepare_components(self, model_config: ModelConfig):
        # Ensure neural architectures
        model_config.branch.component_type = "branch_neural"
        model_config.trunk.component_type = "trunk_neural"

    def setup_training(self, model: 'DeepONet'):
        model.trunk.requires_grad_(True)
        model.branch.requires_grad_(True)
        trainable_params = self._get_trainable_parameters(model)
        if not trainable_params:
            raise ValueError("No trainable parameters found in the model.")
        self.train_schedule = []
        for spec in self.config.optimizer_scheduler: # type: ignore
            if isinstance(spec, dict):
                spec = OptimizerSpec(**spec)
            optimizer = create_optimizer(spec, trainable_params)
            scheduler = create_scheduler(spec, optimizer)
            self.train_schedule.append((spec.epochs, optimizer, scheduler))

    def _get_trainable_parameters(self, model: 'DeepONet'):
        trainable_params = []
        for name, param in model.trunk.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        for name, param in model.branch.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def get_train_schedule(self) -> list[tuple[int, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]]:
        if not hasattr(self, 'train_schedule'):
            raise ValueError("Training schedule not set up. Call setup_training first.")
        return self.train_schedule

    def check_phase_transition(self, epoch: int) -> bool:
        return False  # No phase transitions

    def execute_phase_transition(self, model: 'DeepONet'):
        raise NotImplementedError("Vanilla strategy has no phase transitions")
    
    def validation_enabled(self) -> bool:
        return True
    
    # def on_epoch_end(self, epoch: int, model: 'DeepONet', optimizer: torch.optim.Optimizer,)

    def strategy_specific_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
        relative_error = {'error' : (self.error_metric(y_true - y_pred) / self.error_metric(y_true)).item()}

        return relative_error

    def get_optimizer_scheduler(self):
        return self.config.optimizer_scheduler # type: ignore
    
    def get_phases(self) -> list[str]:
        """Return phase names (e.g., ['phase1', 'phase2'])"""
        return ["vanilla"]

    def apply_gradient_constraints(self, model: DeepONet):
        """Optional gradient clipping/normalization"""
        pass

    def state_dict(self) -> dict:
        """Strategy-specific state for checkpoints"""
        return {}