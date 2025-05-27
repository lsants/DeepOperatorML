from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Dict, Optional, Union, List
from ...model.optimization.optimizers.config import OptimizerConfig
from ...utilities.metrics.errors import ERROR_METRICS
if TYPE_CHECKING:
    from ...model.deeponet import DeepONet
    from ..config import ModelConfig
    from ..optimization.optimizers.config import OptimizerSpec

@dataclass
class StrategyConfig:
    """Base configuration for all training strategies"""
    name: Literal["vanilla", "pod", "two_step"]  # Enforced valid values
    loss: str
    error: str  # Key for ERROR_METRICS lookup
    
    def __post_init__(self):
        self._validate_common()
        
    def _validate_common(self):
        """Validation for base fields only"""
        valid_names = {"vanilla", "pod", "two_step"}
        if self.name not in valid_names:
            raise ValueError(f"Invalid strategy {self.name}. Must be in {valid_names}")
        

@dataclass
class VanillaConfig(StrategyConfig):
    """Configuration for standard end-to-end training strategy."""
    # Inherited from StrategyConfig:
    # name: Literal["vanilla"] (enforced)
    # loss: str
    # error: str
    
    # Vanilla-specific optimization control
    optimizer_scheduler: List[OptimizerSpec]
    
    def __post_init__(self):
        super().__post_init__()
        self._validate_vanilla()
        
    def _validate_vanilla(self):
        pass

@dataclass
class TwoStepConfig(StrategyConfig):
    """Configuration for two-phase training with trunk decomposition."""
    # Inherited from BaseStrategyConfig:
    # name: Literal["two_step"] (enforced)
    # error: str
    
    # Phase control
    trunk_epochs: int  # Phase 1 duration
    branch_epochs: int  # Phase 2 duration
    
    # Optimization
    two_step_optimizer_schedule: Dict[str, List[OptimizerSpec]]  # {"phase1": [...], "phase2": [...]}
    
    # Decomposition
    decomposition_type: Literal["svd", "qr"]  # Enforced decomposition method
    
    def __post_init__(self):
        super().__post_init__()
        self._validate_two_step()
        
    def _validate_two_step(self):
        """TwoStep-specific validation"""
        # Phase epochs
        if self.trunk_epochs <= 0 or self.branch_epochs <= 0:
            raise ValueError("Epoch counts must be > 0")
            
        # Phase optimizer definitions
        if "trunk" not in self.two_step_optimizer_schedule or "branch" not in self.two_step_optimizer_schedule:
            raise ValueError("two_step_optimizer_schedule must define 'branch' and 'trunk' phases.")
        for phase, optims in self.two_step_optimizer_schedule.items():
            for o in optims:
                if "optimizer_type" not in str(o):
                    raise ValueError(f"Optimizer in {phase} missing 'type' key")

@dataclass
class PODConfig(StrategyConfig):
    """Configuration for Proper Orthogonal Decomposition training strategy."""
    num_pod_modes: int  # Required number of POD modes
    pod_basis: torch.Tensor  # Precomputed basis tensor [modes x features]
    optimizer_scheduler: list[OptimizerSpec]
    pod_means: Optional[torch.Tensor] = None  # Optional mean for centering

    def __post_init__(self):
        super().__post_init__()
        self._validate_pod()

    def _validate_pod(self):
        """Strategy-specific validation"""
        if self.num_pod_modes <= 0:
            raise ValueError("POD requires num_pod_modes > 0")
        if self.pod_basis.ndim != 2:
            raise ValueError("pod_basis must be 2D (modes x features)")
        if "optimizer_type" not in self.optimizer_scheduler:
            raise ValueError("POD optimizer config requires 'type' key")
        
StrategyConfigUnion = Union[VanillaConfig, PODConfig, TwoStepConfig]