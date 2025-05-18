import torch
from abc import ABC, abstractmethod
from typing import Any

class BaseBranch(ABC):
    @abstractmethod
    def forward(self, branch_input: Any) -> torch.Tensor:
        """Computes the branch output (coefficients) from the input."""
        pass