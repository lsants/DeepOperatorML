from ..config import ComponentConfig
from .branch_factory import BranchRegistry
from .base import Branch
import torch

@BranchRegistry.register("matrix")
class MatrixBranch(Branch):
    def __init__(self, config: ComponentConfig):
        self.weights = torch.nn.Parameter(torch.randn(config.input_dim, 
                                                config.output_dim))

    def forward(self, x):
        return x @ self.weights