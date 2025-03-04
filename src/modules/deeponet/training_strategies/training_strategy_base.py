from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple, Optional
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

import torch

class TrainingStrategy(ABC):
    def __init__(self, loss_fn: callable) -> None:
        self.loss_fn: callable = loss_fn
        self.inference: bool = False

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
                xb: Optional[torch.Tensor] = None, 
                xt: Optional[torch.Tensor] = None,
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
                     outputs: Tuple[torch.Tensor], 
                     batch: Dict[str, torch.Tensor], 
                     model: 'DeepONet', 
                     params: Dict[str, Any], 
                     **kwargs
    ) -> float:
        """
        Computes the loss for the current batch.
        """
        pass

    @abstractmethod
    def compute_errors(self, 
                       outputs: Tuple[torch.Tensor], 
                       batch: Dict[str, torch.Tensor], 
                       model: 'DeepONet', 
                       params: Dict[str, Any], 
                       **kwargs
    ) -> Dict[str, Any]:
        """
        Computes error metrics for evaluation.
        """
        pass

    @abstractmethod
    def get_branch_config(self, base_branch_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns an updated branch configuration dictionary.
        This method merges strategy-specific parameters into the base_branch_config.
        """
        pass

    @abstractmethod
    def get_trunk_config(self, base_trunk_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns an updated trunk configuration dictionary.
        This method merges strategy-specific parameters into the base_trunk_config.
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

    def after_epoch(self, epoch: int, model, params: Dict[str, Any], **kwargs) -> None:
        pass