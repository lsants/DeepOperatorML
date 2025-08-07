from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from src.modules.models.tools.optimizers.config import OptimizerSpec

@dataclass
class StrategyConfig:
    """
    Base configuration for all training strategies.

    This dataclass defines the common fields that all strategies share,
    along with validation logic to ensure the strategy name is recognized.
    """
    name: Literal["vanilla", "pod", "two_step"]
    error: str

    def __post_init__(self):
        """Perform validation after initialization."""
        self._validate_common()

    def _validate_common(self):
        """Validation for common fields across all strategies."""
        valid_names = {"vanilla", "pod", "two_step"}
        if self.name not in valid_names:
            raise ValueError(
                f"Invalid strategy {self.name}. Must be in {valid_names}")
    @classmethod
    def setup_for_training(cls, train_cfg: dict, data_cfg: dict) -> 'StrategyConfig':
        """
        Creates a StrategyConfig instance from training and data dictionaries.

        This factory method is useful for setting up the configuration at the
        start of a training run, extracting the necessary information from
        larger configuration files.

        Args:
            train_cfg (dict): The training configuration dictionary.
            data_cfg (dict): The data configuration dictionary.

        Returns:
            StrategyConfig: A base strategy configuration object.
        """
        name = data_cfg["shapes"][data_cfg["targets"][0]][-1]
        error = train_cfg["output_handling"]
        return cls(
            error=error,
            name=name
        )
    @classmethod
    def setup_for_inference(cls, model_cfg_dict) -> 'StrategyConfig':
        """
        Creates a StrategyConfig instance from a saved model configuration.

        This method is used to reconstruct the strategy configuration from
        a dictionary loaded from a checkpoint or a saved model file.

        Args:
            model_cfg_dict (dict): The dictionary containing the saved model configuration.

        Returns:
            StrategyConfig: A base strategy configuration object.
        """
        name = model_cfg_dict["strategy"]["name"]
        error = model_cfg_dict["strategy"]["error"]
        return cls(
            error=error,
            name=name
        )
@dataclass
class VanillaConfig(StrategyConfig):
    """
    Configuration for the standard end-to-end training strategy.

    This strategy trains the entire DeepONet model in one phase without
    any special decomposition or multi-step training.
    """
    loss: str
    optimizer_scheduler: list[OptimizerSpec]

    def __post_init__(self):
        """Performs validation after initialization, including base checks."""
        super().__post_init__()
        self._validate_vanilla()

    def _validate_vanilla(self):
        """Vanilla-specific validation checks."""
        pass

@dataclass
class TwoStepConfig(StrategyConfig):
    """
    Configuration for a two-phase training strategy with trunk decomposition.

    This strategy first trains the trunk component and then
    transitions to training the branch. This assures better stability during optimization.
    """
    loss: str
    two_step_optimizer_schedule: dict[str, list[OptimizerSpec]]
    decomposition_type: Literal["svd", "qr"]  # Enforced decomposition method
    num_branch_train_samples: int

    def __post_init__(self):
        """Performs validation after initialization, including base checks."""
        super().__post_init__()
        self._validate_two_step()

    def _validate_two_step(self):
        """TwoStep-specific validation."""
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
    """
    Configuration for the Proper Orthogonal Decomposition (POD) training strategy.

    This strategy uses a pre-computed POD basis to represent the solution
    space (trunk).
    """
    loss: str
    pod_basis: torch.Tensor
    pod_type: Literal['stacked', 'by_channel']
    optimizer_scheduler: list[OptimizerSpec]

    def __post_init__(self):
        """Performs validation after initialization, including base checks."""
        super().__post_init__()
        self._validate_pod()

    def _validate_pod(self):
        """POD-specific validation checks."""
        pass