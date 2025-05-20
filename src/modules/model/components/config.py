from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class ComponentConfig:
    architecture: str
    layers: list[int] = None # Specific to neural networks
    activation: str = None # Specific to Universal Approximation Theorem-based architectures
    degree: int = None # KAN-specific
    params: dict[str, Any] = None

    def validate_architecture(self):
        valid_archs = {'resnet', 'kan', 'pod', 'mlp'}
        if self.architecture not in valid_archs:
            raise ValueError(f"Invalid architecture {self.architecture}")
    # add a POD-specific?



    