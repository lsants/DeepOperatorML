from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Dict, Union, List
if TYPE_CHECKING:
    from ..optimization.optimizers.config import OptimizerSpec


@dataclass
class StrategyConfig:
    """Base configuration for all training strategies"""
    name: Literal["vanilla", "pod", "two_step"]
    loss: str
    error: str
    epochs: int

    def __post_init__(self):
        self._validate_common()

    def _validate_common(self):
        """Validation for base fields only"""
        valid_names = {"vanilla", "pod", "two_step"}
        if self.name not in valid_names:
            raise ValueError(
                f"Invalid strategy {self.name}. Must be in {valid_names}")


@dataclass
class VanillaConfig(StrategyConfig):
    """Configuration for standard end-to-end training strategy."""
    optimizer_scheduler: List[OptimizerSpec]

    def __post_init__(self):
        super().__post_init__()
        self._validate_vanilla()

    def _validate_vanilla(self):
        pass


@dataclass
class TwoStepConfig(StrategyConfig):
    """Configuration for two-phase training with trunk decomposition."""
    trunk_epochs: int
    branch_epochs: int
    two_step_optimizer_schedule: Dict[str, List[OptimizerSpec]]
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
        if "trunk_phase" not in self.two_step_optimizer_schedule or "branch_phase" not in self.two_step_optimizer_schedule:
            raise ValueError(
                "two_step_optimizer_schedule must define 'branch' and 'trunk' phases.")
        for phase, optims in self.two_step_optimizer_schedule.items():
            for o in optims:
                if "optimizer_type" not in str(o):
                    raise ValueError(
                        f"Optimizer in {phase} missing 'type' key")


@dataclass
class PODConfig(StrategyConfig):
    """Configuration for Proper Orthogonal Decomposition training strategy."""
    pod_basis: torch.Tensor
    pod_mean: torch.Tensor
    optimizer_scheduler: list[OptimizerSpec]

    def __post_init__(self):
        super().__post_init__()
        self._validate_pod()

    def _validate_pod(self):
        pass


StrategyConfigUnion = Union[VanillaConfig, PODConfig, TwoStepConfig]
