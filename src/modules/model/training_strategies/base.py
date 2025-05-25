from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Dict
if TYPE_CHECKING:
    from ...model.deeponet import DeepONet
    from ..config import ModelConfig

@dataclass
class StrategyConfig:
    # Common to ALL strategies
    name: str  # "vanilla", "pod", "two_step"
    basis_functions: int
    
    # POD-specific (only used when name="pod")
    num_pod_modes: Optional[float] = None
    pod_basis: Optional[torch.Tensor] = None
    pod_means: Optional[torch.Tensor] = None
    
    # TwoStep-specific (only used when name="two_step")
    two_step_phase_optimizers: Optional[Dict[str, List[dict]]] = None
    two_step_trunk_epochs: Optional[int] = None
    two_step_branch_epochs: Optional[int] = None
    decomposition_type: Optional[str] = None
    
    # Vanilla-specific
    global_schedule: Optional[List[dict]] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Enforce strategy-specific parameter requirements"""

        if self.name == "two_step":
            if not self.two_step_phase_optimizers:
                raise ValueError("TwoStep requires phase_optimizers")
            if not self.two_step_trunk_epochs:
                raise ValueError("TwoStep requires trunk_epochs")

        elif self.name == "vanilla":
            if not self.global_schedule:
                raise ValueError("Vanilla requires global_schedule")

class TrainingStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    @abstractmethod
    def prepare_components(self, model_config: ModelConfig): ...

    @abstractmethod
    def setup_training(self, model: DeepONet): ...
    
    @abstractmethod
    def check_phase_transition(self, model: DeepONet, epoch: int) -> bool: ...
    
    @abstractmethod
    def execute_phase_transition(self, model: DeepONet): ...