from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Callable
from ..registry import ComponentRegistry


@dataclass
class BranchConfig:
    # Fundamental component type
    architecture: Optional[Literal["resnet",
                                   "mlp",
                                   "chebyshev_kan",
                                   "trainable_matrix"]]
    component_type: Literal["neural_branch", "matrix_branch"] = "neural_branch"
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    # Architecture-specific parameters (for neural)
    hidden_layers: Optional[list[int]] = None
    dropout_rates: Optional[list[float]] = None
    activation: Optional[Callable | str] = None
    degree: Optional[int] = None
    # Matrix-specific parameters
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None


class BranchConfigValidator:
    @staticmethod
    def validate(config: BranchConfig):
        try:
            component_class, required_params = ComponentRegistry.get(
                component_type=config.component_type,
                architecture=config.architecture
            )

            required_params = [p for p in required_params if p != "self"]

            # Check for missing parameters
            missing = [p for p in required_params if not hasattr(config, p)]
            if missing:
                raise ValueError(
                    f"Missing required parameters for {config.architecture}: {missing}"
                )

        except ValueError as e:
            raise ValueError(f"Invalid branch configuration: {e}") from e

        # Architecture-specific validation
        if 'kan' in str(config.architecture) and config.degree < 1:  # type: ignore
            raise ValueError("KAN requires degree >= 1")
