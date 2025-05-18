from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from ...data_processing.transforms import Compose
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from modules.model.deeponet import DeepONet


class TrainingStrategy(ABC):
    def __init__(self, loss_fn: Callable[[Iterable[torch.Tensor], Iterable[torch.Tensor]], 
                                         torch.Tensor]) -> None:
        self.loss_fn = loss_fn
        self.inference: bool = False

    @abstractmethod
    def prepare_for_training(self, model: 'DeepONet') -> None:
        """
        Prepares the model for training.
        This method should handle any one-time model adjustments, resource allocations,
        or configuration steps needed before training starts.
        """
        pass

    @abstractmethod
    def forward(self,
                model: 'DeepONet',
                branch_input: torch.Tensor | None = None,
                trunk_input: torch.Tensor | None = None,
                **kwargs: Any
                ) -> tuple[torch.Tensor]:
        """
        Implements the forward pass for the training strategy.
        Concrete implementations should determine how the model processes the inputs,
        including any custom behavior (e.g. matrix-based operations or phased training).
        """
        pass

    @abstractmethod
    def compute_loss(self,
                     outputs: tuple[torch.Tensor],
                     batch: dict[str, torch.Tensor],
                     model: 'DeepONet',
                     training_params: dict[str, Any],
                     **kwargs
                     ) -> torch.Tensor:
        """
        Computes the loss for the current batch.
        """
        pass

    @abstractmethod
    def compute_errors(self,
                       outputs: tuple[torch.Tensor],
                       batch: dict[str, torch.Tensor],
                       model: 'DeepONet',
                       training_params: dict[str, Any],
                       **kwargs
                       ) -> dict[str, Any]:
        """
        Computes error metrics for evaluation.
        """
        pass

    @abstractmethod
    def get_branch_config(self, branch_config: dict[str, Any]) -> dict[str, Any]:
        """
        Returns an updated branch configuration dictionary.
        This method merges strategy-specific parameters into the branch_config.
        """
        pass

    @abstractmethod
    def get_trunk_config(self, trunk_config: dict[str, Any]) -> dict[str, Any]:
        """
        Returns an updated trunk configuration dictionary.
        This method merges strategy-specific parameters into the trunk_config.
        """
        pass

    def inference_mode(self) -> None:
        """
        Puts the strategy into inference mode.
        """
        self.inference = True

        # For strategies that use multiple phases (e.g., two-step), you may also define:
    def update_training_phase(self, phase: str) -> None:
        pass

    def prepare_for_phase(self, model, **kwargs) -> None:
        pass

    def after_epoch(self, epoch: int, model, params: dict[str, Any], **kwargs) -> None:
        pass
