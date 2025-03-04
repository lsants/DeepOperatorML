import torch
from abc import ABC, abstractmethod
from base_branch import BaseBranch

class BaseBranch(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the branch output (coefficients) from the input."""
        pass