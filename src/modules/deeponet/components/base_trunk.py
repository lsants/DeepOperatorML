import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTrunk(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the trunk output (basis functions) from the input."""
        pass

    @abstractmethod
    def get_basis(self) -> torch.Tensor:
        """Returns the basis functions for inference.
           For a trainable trunk, this might be computed on the fly;
           for a fixed trunk, this simply returns the stored tensor.
        """
        pass