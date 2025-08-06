from __future__ import annotations
import torch
from src.modules.models.tools.optimizers.config import OptimizerSpec
from src.modules.models.deeponet.training_strategies.base import TrainingStrategy
from src.modules.models.tools.metrics.errors import ERROR_METRICS
from src.modules.models.tools.optimizers.config import OptimizerSpec
from src.modules.models.tools.optimizers.optimizer_factory import create_optimizer, create_scheduler
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig
    from src.modules.models.deeponet.deeponet import DeepONet
    from src.modules.models.deeponet.training_strategies.config import VanillaConfig


class VanillaStrategy(TrainingStrategy):
    def __init__(self, config: VanillaConfig):
        super().__init__(config)
        self.error_metric = ERROR_METRICS[config.error.lower()]

    def prepare_components(self, model_config: DeepONetConfig):
        model_config.branch.component_type = "neural_branch"
        model_config.trunk.component_type = "neural_trunk"

    def setup_training(self, model: 'DeepONet'):
        model.trunk.requires_grad_(True)
        model.branch.requires_grad_(True)
        model.bias.requires_grad_(True)
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
        for name, param in model.trunk.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        for name, param in model.branch.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        for name, param in model.bias.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def get_train_schedule(self) -> list[tuple[int, torch.optim.optimizer.Optimizer, torch.optim.lr_scheduler._LRScheduler]]:
        if not hasattr(self, 'train_schedule'):
            raise ValueError(
                "Training schedule not set up. Call setup_training first.")
        return self.train_schedule

    def execute_phase_transition(self, model: 'DeepONet'):
        raise NotImplementedError("Vanilla strategy has no phase transitions")

    def validation_enabled(self) -> bool:
        return True

    def strategy_specific_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, label_map: list[str] | None = None) -> dict[str, float]:
        relative_error = self.error_metric(
            y_true - y_pred) / self.error_metric(y_true)
        if label_map is not None:
            strategy_metric = {
                **{f'Error_{label_map[i]}': e.item() for i, e in enumerate(relative_error.detach())}
            }
        else:
            strategy_metric = {
                **{f'Error_{i}': e.item() for i, e in enumerate(relative_error.detach())}
            }
        return strategy_metric

    def get_optimizer_scheduler(self):
        return self.config.optimizer_scheduler  # type: ignore

    def get_phases(self) -> list[str]:
        """Return phase names (e.g., ['phase1', 'phase2'])"""
        return ["vanilla"]

    def apply_gradient_constraints(self, model: DeepONet):
        """Optional gradient clipping/normalization"""
        pass

    def state_dict(self) -> dict:
        """Strategy-specific state for checkpoints"""
        return {}

    def should_transition_phase(self, current_phase: int, current_epoch: int) -> bool:
        return False
