from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union
if TYPE_CHECKING:
    from ..optimization.optimizers.config import OptimizerSpec


@dataclass
class StrategyConfig:
    """Base configuration for all training strategies"""
    name: Literal["vanilla", "pod", "two_step"]
    error: str

    def __post_init__(self):
        self._validate_common()

    def _validate_common(self):
        """Validation for base fields only"""
        valid_names = {"vanilla", "pod", "two_step"}
        if self.name not in valid_names:
            raise ValueError(
                f"Invalid strategy {self.name}. Must be in {valid_names}")
    @classmethod
    def setup_for_training(cls, train_cfg: dict, data_cfg: dict):
        name = data_cfg["shapes"][data_cfg["targets"][0]][-1]
        error = train_cfg["output_handling"]
        return cls(
            error=error,
            name=name
        )
    @classmethod
    def setup_for_inference(cls, model_cfg_dict):
        name = model_cfg_dict["strategy"]["name"]
        error = model_cfg_dict["strategy"]["error"]
        return cls(
            error=error,
            name=name
        )
@dataclass
class VanillaConfig(StrategyConfig):
    """Configuration for standard end-to-end training strategy."""
    loss: str
    optimizer_scheduler: list[OptimizerSpec]

    def __post_init__(self):
        super().__post_init__()
        self._validate_vanilla()

    def _validate_vanilla(self):
        pass


@dataclass
class TwoStepConfig(StrategyConfig):
    """Configuration for two-phase training with trunk decomposition."""
    loss: str
    two_step_optimizer_schedule: dict[str, list[OptimizerSpec]]
    decomposition_type: Literal["svd", "qr"]  # Enforced decomposition method
    num_branch_train_samples: int

    def __post_init__(self):
        super().__post_init__()
        self._validate_two_step()

    def _validate_two_step(self):
        """TwoStep-specific validation"""
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
    loss: str
    pod_basis: torch.Tensor
    optimizer_scheduler: list[OptimizerSpec]

    def __post_init__(self):
        super().__post_init__()
        self._validate_pod()

    def _validate_pod(self):
        pass


StrategyConfigUnion = Union[VanillaConfig, PODConfig, TwoStepConfig]
