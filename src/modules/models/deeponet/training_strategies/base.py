from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, Iterable
from src.modules.models.tools.loss_functions.get_loss_function import get_loss_function
from src.modules.models.tools.metrics.errors import ERROR_METRICS
if TYPE_CHECKING:
    from src.modules.models.deeponet.deeponet import DeepONet
    from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig
    from src.modules.models.deeponet.training_strategies.config import StrategyConfig


class TrainingStrategy(ABC):
    """
    An abstract base class (ABC) that defines the interface for a training strategy.

    This class serves as a blueprint for implementing different training loops,
    allowing for flexible and specialized training methodologies for DeepONet
    models. Concrete implementations must provide logic for preparing components,
    defining training phases, handling gradients, and calculating metrics.
    """
    def __init__(self, config: StrategyConfig):
        """
        Initializes the TrainingStrategy with a configuration.

        Args:
            config (StrategyConfig): The configuration object for the strategy,
                                     specifying the loss and error metrics to use.
        """
        self.config = config
        self.epoch_count = 0
        self.error_metric = ERROR_METRICS[config.error.lower()]
        self.loss = self.get_criterion()

    @abstractmethod
    def prepare_components(self, model_config: DeepONetConfig):
        """
        Modifies the components configuration for the strategy before model initialization.

        This method allows a strategy to dynamically change the model architecture
        or other settings based on its specific requirements before the model
        is instantiated.
        """
        pass

    @abstractmethod
    def get_phases(self) -> list[str]:
        """
        Returns a list of phase names for the strategy.

        This defines the logical steps of a multi-phase training strategy,
        such as "pre-training" or "fine-tuning".

        Returns:
            list[str]: A list of phase names (e.g., ['phase1', 'phase2']).
        """
        pass

    @abstractmethod
    def apply_gradient_constraints(self, model: DeepONet):
        """
        Applies optional gradient clipping or normalization.

        This method is called during the training loop to apply constraints
        to the model's gradients before the optimizer step.
        """
        pass

    @abstractmethod
    def setup_training(self, model: DeepONet):
        """
        Initializes the optimizer and other training-specific setups.

        This method should configure the optimizers and any other necessary
        components for the training process based on the strategy.
        """
        pass

    @abstractmethod
    def get_train_schedule(self) -> list[tuple[int, torch.optim.optimizer.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]]:
        """
        Returns the training schedule for the strategy.

        Each tuple in the list should contain:
        1. The number of epochs for a phase.
        2. The optimizer for that phase.
        3. An optional learning rate scheduler for that phase.

        Returns:
            list[tuple[int, torch.optim.optimizer.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]]:
                The training schedule.
        """
        pass

    @abstractmethod
    def execute_phase_transition(self, model: DeepONet, all_branch_indices: Optional[torch.Tensor] = None, full_trunk_batch: Optional[torch.Tensor] = None, full_outputs_batch: Optional[torch.Tensor] = None):
        """
        Performs the actual phase transition.

        This is where a strategy can implement logic like decomposing the trunk
        or updating components to prepare for the next training phase.
        """
        pass

    @abstractmethod
    def validation_enabled(self) -> bool:
        """
        Indicates whether to use a validation set during training.

        Returns:
            bool: True if validation is enabled, False otherwise.
        """
        pass

    @abstractmethod
    def strategy_specific_metrics(self,
                                  y_true: torch.Tensor,
                                  y_pred: torch.Tensor,
                                  branch_input: torch.Tensor | None = None,
                                  trunk_input: torch.Tensor | None = None,
                                  label_map: list[str] | None = None) -> dict[str, float]:
        """
        Computes strategy-specific metrics during training.

        This allows for custom metrics that are relevant only to a particular
        training strategy.

        Args:
            y_true (torch.Tensor): The ground truth target values.
            y_pred (torch.Tensor): The predicted target values from the model.
            branch_input (Optional[torch.Tensor]): The branch input tensor.
            trunk_input (Optional[torch.Tensor]): The trunk input tensor.
            label_map (Optional[List[str]]): A list of labels for each output dimension.

        Returns:
            dict[str, float]: A dictionary of metric names and their values.
        """
        pass

    @abstractmethod
    def get_optimizer_scheduler(self): 
        """
        A placeholder for a method to get the optimizer and scheduler.

        This method is marked abstract to ensure all concrete implementations
        define how they create their optimizer and scheduler.
        """
        ...

    def state_dict(self) -> dict:
        """
        Returns the strategy-specific state for checkpoints.

        This method is used to save the state of the training strategy for
        resuming training later. The default implementation returns an
        empty dictionary.

        Returns:
            dict: A dictionary containing the state of the strategy.
        """
        return {}

    def get_criterion(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Creates the loss function based on the strategy's configuration.

        This provides a default implementation for creating the loss function,
        which can be overridden if a strategy requires a custom loss.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: The loss function.
        """
        return get_loss_function(name=self.config.loss) # type: ignore

    def should_transition_phase(self, current_phase: int, current_epoch: int) -> bool:
        """
        Determines if the strategy should transition to a new phase.

        Returns:
            bool: True if a phase transition is needed, False otherwise.
        """
        return False

    def compute_loss(self, model: DeepONet,
                     x_branch: torch.Tensor,
                     x_trunk: torch.Tensor,
                     y_true: torch.Tensor,
                     indices: tuple[Iterable[int], ...]) -> tuple[torch.Tensor, ...]:
        """
        Computes the loss for the given model and data.

        Args:
            model (DeepONet): The model being trained.
            x_branch (torch.Tensor): The branch input tensor.
            x_trunk (torch.Tensor): The trunk input tensor.
            y_true (torch.Tensor): The ground truth target values.
            indices (Tuple[Iterable[int], ...]): The indices for the current batch.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing the model's predictions
                                       and the computed loss.
        """
        y_pred = model(x_branch, x_trunk)
        loss = self.loss(y_pred, y_true)
        return y_pred, loss

    def calculate_metrics(
            self,
            model: torch.nn.Module,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            loss: float,
            train: bool,
            branch_indices: Iterable[int] | None = None,
            trunk_input: torch.Tensor | None = None,
            label_map: list[str] | None = None
            ) -> dict[str, float]:
        """
        Combines base and strategy-specific metrics.

        Args:
            model (torch.nn.Module): The model being evaluated.
            y_true (torch.Tensor): The ground truth target values.
            y_pred (torch.Tensor): The predicted target values.
            loss (float): The loss value for the current batch.
            train (bool): A flag indicating if this is a training step.
            branch_indices (Optional[Iterable[int]]): Indices of the branch data.
            trunk_input (Optional[torch.Tensor]): The trunk input data.
            label_map (Optional[List[str]]): A list of labels for each output dimension.

        Returns:
            dict[str, float]: A dictionary of combined metrics.
        """
        metrics = self.base_metrics(y_true, y_pred, loss, label_map=label_map)
        metrics.update(
            self.strategy_specific_metrics(
                y_true=y_true, y_pred=y_pred, label_map=label_map)
        )
        return metrics

    def base_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, loss: float, label_map: list[str] | None = None) -> dict[str, float]:
        """
        Computes common metrics for all strategies.

        Args:
            y_true (torch.Tensor): The ground truth target values.
            y_pred (torch.Tensor): The predicted target values.
            loss (float): The loss value for the current batch.
            label_map (Optional[List[str]]): A list of labels for each output dimension.

        Returns:
            dict[str, float]: A dictionary of common metrics.
        """
        with torch.no_grad():
            error = self.error_metric(y_pred - y_true)
        if label_map is not None:
            base_metric = {
                'loss': loss,
                **{f'Error_{label_map[i]}': e.item() for i, e in enumerate(error.detach())}
            }
        else:
            base_metric = {
                'loss': loss,
                **{f'Error_{i}': e.item() for i, e in enumerate(error.detach())}
            }
        return base_metric
