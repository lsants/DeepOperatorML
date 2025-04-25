import torch
from abc import ABC, abstractmethod
from ...data_processing.transforms import Compose
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet


class TrainingStrategy(ABC):
    def __init__(self, loss_fn: callable, output_transform: Compose | None = None) -> None:
        self.loss_fn: callable = loss_fn
        self.inference: bool = False
        self.output_transform: Compose | None = output_transform

    @abstractmethod
    def prepare_training(self, model: 'DeepONet', **kwargs) -> None:
        """
        Prepares the model for training.
        This method should handle any one-time model adjustments, resource allocations,
        or configuration steps needed before training starts.
        """
        pass

    @abstractmethod
    def forward(self,
                model: 'DeepONet',
                xb: torch.Tensor = None,
                xt: torch.Tensor = None,
                **kwargs
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
                     params: dict[str, any],
                     **kwargs
                     ) -> float:
        """
        Computes the loss for the current batch.
        """
        pass

    @abstractmethod
    def compute_errors(self,
                       outputs: tuple[torch.Tensor],
                       batch: dict[str, torch.Tensor],
                       model: 'DeepONet',
                       params: dict[str, any],
                       **kwargs
                       ) -> dict[str, any]:
        """
        Computes error metrics for evaluation.
        """
        pass

    @abstractmethod
    def get_branch_config(self, branch_config: dict[str, any]) -> dict[str, any]:
        """
        Returns an updated branch configuration dictionary.
        This method merges strategy-specific parameters into the branch_config.
        """
        pass

    @abstractmethod
    def get_trunk_config(self, trunk_config: dict[str, any]) -> dict[str, any]:
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

    def after_epoch(self, epoch: int, model, params: dict[str, any], **kwargs) -> None:
        pass
