from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from ..optimization.loss_functions.get_loss_function import get_loss_function
from ...utilities.metrics.errors import ERROR_METRICS
if TYPE_CHECKING:
    from ...model.deeponet import DeepONet
    from ..config import ModelConfig
    from .config import StrategyConfig


class TrainingStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.epochs = config.epochs
        self.epoch_count = 0
        self.error_metric = ERROR_METRICS[config.error.lower()]
        self.loss = self.get_criterion()

    @abstractmethod
    def prepare_components(self, model_config: ModelConfig):
        """Modifies the components configuration for the strategy before model initialization.
        """
        pass

    @abstractmethod
    def get_phases(self) -> List[str]:
        """Return phase names (e.g., ['phase1', 'phase2'])"""
        pass

    @abstractmethod
    def apply_gradient_constraints(self, model: DeepONet):
        """Optional gradient clipping/normalization"""
        pass

    @abstractmethod
    def setup_training(self, model: DeepONet):
        """Initialize optimizer based on the strategy.
        """
        pass

    @abstractmethod
    def get_train_schedule(self) -> List[tuple[int, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]]:
        """Returns the training schedule for the strategy.
        Each tuple contains (epochs, optimizer, scheduler).
        """
        pass

    @abstractmethod
    def execute_phase_transition(self, model: DeepONet, full_branch_batch: Optional[torch.Tensor] = None, full_trunk_batch: Optional[torch.Tensor] = None):
        """Perform the actual phase transition, such as decomposing the trunk or updating components.
        """
        pass

    @abstractmethod
    def validation_enabled(self) -> bool:
        """Whether to use validation set during training"""
        pass

    @abstractmethod
    def strategy_specific_metrics(self,
                                  y_true: torch.Tensor,
                                  y_pred: torch.Tensor,
                                  branch_input: torch.Tensor | None = None,
                                  trunk_input: torch.Tensor | None = None) -> dict[str, float]:
        """Strategy-specific metrics to be computed during training"""
        pass

    @abstractmethod
    def get_optimizer_scheduler(self): ...

    def training_complete(self) -> bool:
        self.epoch_count += 1
        return self.epoch_count > self.epochs

    def state_dict(self) -> Dict:
        """Strategy-specific state for checkpoints"""
        return {}

    def get_criterion(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Default implementation using strategy's config"""
        return get_loss_function(name=self.config.loss)

    def should_transition_phase(self, current_phase: int, current_epoch: int) -> bool:
        """Return True if the strategy should transition to a new phase."""
        return False

    def compute_loss(self, model: DeepONet,
                     x_branch: torch.Tensor,
                     x_trunk: torch.Tensor,
                     y_true: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Computes the loss for the given model and data"""
        y_pred = model(x_branch, x_trunk)
        loss = self.loss(y_pred, y_true)
        return y_pred, loss

    def calculate_metrics(self,
                          y_true: torch.Tensor,
                          y_pred: torch.Tensor,
                          loss: float,
                          train: bool,
                          branch_input: torch.Tensor | None = None,
                          trunk_input: torch.Tensor | None = None) -> Dict[str, float]:
        """Combines base and strategy-specific metrics"""
        metrics = self.base_metrics(y_true, y_pred, loss)
        metrics.update(
            self.strategy_specific_metrics(y_true=y_true, y_pred=y_pred)
        )
        return metrics

    def base_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, loss: float) -> Dict[str, float]:
        """Common metrics for all strategies"""
        with torch.no_grad():
            error = self.error_metric(y_pred - y_true).item()
        return {
            'loss': loss,
            'error': error
        }
