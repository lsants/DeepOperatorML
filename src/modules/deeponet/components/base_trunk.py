import torch
from abc import ABC, abstractmethod
from typing import Any
class BaseTrunk(ABC):
    @abstractmethod
    def forward(self, trunk_input: Any ) -> torch.Tensor:
        """Computes the trunk output (basis functions) from the input."""
        pass

    @abstractmethod
    def get_basis(self) -> torch.Tensor:
        """Returns the basis functions for inference.
           For a trainable trunk, this might be computed on the fly;
           for a fixed trunk, this simply returns the stored tensor.
        """
        pass    