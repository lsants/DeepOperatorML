from __future__ import annotations
import torch
import logging
from typing import TYPE_CHECKING, Any
from src.modules.models.deeponet.training_strategies.config import PODConfig
from src.modules.models.deeponet.training_strategies.base import TrainingStrategy
from src.modules.models.tools.metrics.errors import ERROR_METRICS
from src.modules.models.tools.optimizers.config import OptimizerSpec
from src.modules.models.tools.optimizers.optimizer_factory import create_optimizer, create_scheduler
from src.modules.models.deeponet.components.trunk import PODTrunk

if TYPE_CHECKING:
    from ..deeponet import DeepONet
    from ..config.deeponet_config import DeepONetConfig

logger = logging.getLogger(name=__name__)


class PODStrategy(TrainingStrategy):
    """
    Implements a training strategy for DeepONet using Proper Orthogonal Decomposition (POD).

    This strategy assumes the trunk network is replaced by a fixed POD basis
    and only the branch network is trained. The core idea is to project the
    problem onto a low-dimensional basis to simplify the learning task.
    """
    def __init__(self, config: 'PODConfig'):
        """
        Initializes the POD strategy with its specific configuration.

        Args:
            config (PODConfig): The configuration object for the POD strategy.
        """
        super().__init__(config)
        self.error_metric = ERROR_METRICS[config.error.lower()]
        self.pod_type = config.pod_type

    def prepare_components(self, model_config: 'DeepONetConfig'):
        """
        Modifies the model configuration to set up the POD-specific components.

        This method configures the DeepONet to use a 'neural_branch' and a
        'pod_trunk', and sets the pod_basis and embedding dimension based on the
        provided configuration.

        Args:
            model_config (DeepONetConfig): The configuration for the DeepONet model.
        """
        if not isinstance(self.config, PODConfig):
            raise TypeError("PODStrategy requires PODConfig")
        model_config.branch.component_type = "neural_branch"
        model_config.trunk.component_type = "pod_trunk"
        model_config.trunk.architecture = "precomputed"
        model_config.trunk.pod_basis = self.config.pod_basis
        model_config.rescaling.embedding_dimension = self.config.pod_basis.shape[-1]
        if model_config.output.handler_type == 'shared_branch':
            model_config.rescaling.embedding_dimension = self.config.pod_basis.shape[-1] // model_config.output.num_channels
            model_config.trunk.output_dim = model_config.rescaling.embedding_dimension
        elif model_config.output.handler_type == 'split_outputs':
            model_config.rescaling.embedding_dimension = self.config.pod_basis.shape[-1] // model_config.output.num_channels
            model_config.trunk.output_dim = model_config.rescaling.embedding_dimension

    def setup_training(self, model: 'DeepONet'):
        """
        Sets up the training environment for the POD strategy.

        This method freezes the trunk, unfreezes the branch, and configures
        the optimizer and learning rate scheduler based on the strategy's config.

        Args:
            model (DeepONet): The DeepONet model to be trained.
        """
        if not isinstance(model.trunk, PODTrunk):
            raise TypeError(
                "Model's trunk was not correctly defined as 'PODTrunk'.")

        model.trunk.requires_grad_(False)
        model.branch.requires_grad_(True)

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

    def _get_trainable_parameters(self, model: 'DeepONet') -> list[Any]:
        """
        Identifies and returns the trainable parameters of the model.

        For the POD strategy, this will only be the parameters of the branch network.

        Args:
            model (DeepONet): The DeepONet model.

        Returns:
            list: A list of trainable parameters.
        """
        trainable_params = []
        for name, param in model.branch.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def get_train_schedule(self) -> list[tuple[torch.optim.optimizer.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]]:
        """
        Returns the prepared training schedule.

        Returns:
            list: A list of tuples, each containing (epochs, optimizer, scheduler).
        """
        if not hasattr(self, 'train_schedule'):
            raise ValueError(
                "Training schedule not set up. Call setup_training first.")
        return self.train_schedule

    def get_phases(self) -> list[str]:
        """
        Returns the phase names for the POD strategy.

        Since this is a single-phase strategy, it returns a list with one name.

        Returns:
            list[str]: A list containing the phase name, e.g., ["POD"].
        """
        return ["POD"]

    def apply_gradient_constraints(self, model: 'DeepONet'):
        """Optional gradient clipping/normalization"""
        pass

    def execute_phase_transition(self, model: 'DeepONet'):
        """
        Executes a phase transition. Not applicable for this single-phase strategy.
        """
        raise NotImplementedError("POD strategy has no phase transitions")

    def validation_enabled(self) -> bool:
        """
        Checks if validation is enabled. It is enabled by default for this strategy.
        """
        return True

    def strategy_specific_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, label_map: list[str]) -> dict[str, float]:
        """
        Calculates strategy-specific metrics, such as relative error.

        Args:
            y_true (torch.Tensor): The ground truth values.
            y_pred (torch.Tensor): The predicted values.
            label_map (list[str]): The labels for each output dimension.

        Returns:
            dict[str, float]: A dictionary of the calculated metrics.
        """
        relative_error = (self.error_metric(
            y_true - y_pred) / self.error_metric(y_true))
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
        """
        Returns the optimizer and scheduler from the configuration.
        """
        return self.config.optimizer_scheduler # type: ignore

    def should_transition_phase(self, current_phase: int, current_epoch: int) -> bool:
        """
        Determines if a phase transition is needed. Not applicable here.
        """
        return False
