import torch
from .trainable_trunk import TrainableTrunk

class TwoStepTrunk(torch.nn.Module):
    def __init__(self, trunk: TrainableTrunk, A: torch.Tensor) -> None:
        super().__init__()
        self.trunk = trunk
        self.A = torch.nn.Parameter(A)
        torch.nn.init.kaiming_uniform_(self.A)

    def forward(self, trunk_input: torch.Tensor) -> torch.Tensor:
        return self.trunk(trunk_input)