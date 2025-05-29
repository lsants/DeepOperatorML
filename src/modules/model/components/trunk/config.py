from __future__ import annotations
import torch
from typing import Literal, Optional, Callable
from dataclasses import dataclass
from ..registry import ComponentRegistry
from .....exceptions import ConfigValidationError

@dataclass
class TrunkConfig:
    component_type: Literal["trunk_neural", "pod", "decomposed"] = "trunk_neural"
    architecture: Optional[Literal["resnet", "mlp", "chebyshev_kan"]] = None  # Only for neural
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    # Neural architecture params
    hidden_layers: Optional[list[int]] = None
    dropout_rates: Optional[list[float]] = None
    activation: Optional[Callable | str] = None    
    degree: Optional[int] = None
    # POD/Decomposed params
    basis: Optional[torch.Tensor] = None
    num_modes: Optional[float] = None


class TrunkConfigValidator:
    @staticmethod
    def validate(config: TrunkConfig):
        try:
            component_class, required_params = ComponentRegistry.get(
                component_type=config.component_type,
                architecture=config.architecture
            )
            
            # Remove "self" from required parameters
            required_params = [p for p in required_params if p != "self"]
            # Check for missing parameters
            missing = [p for p in required_params if not hasattr(config, p)]
            if missing:
                raise ConfigValidationError(
                    f"Missing required parameters for {config.architecture}: {missing}"
                )
                
        except ValueError as e:
            raise ConfigValidationError(f"Invalid branch configuration: {e}") from e

        # Architecture-specific validation
        if 'kan' in str(config.architecture) and config.degree < 1: # type: ignore
            raise ConfigValidationError("KAN requires degree >= 1")
