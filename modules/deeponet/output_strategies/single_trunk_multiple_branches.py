import torch
from .output_handling_base import OutputHandlingStrategy

class SingleTrunkMultipleBranchesStrategy(OutputHandlingStrategy):
    """Using a single set of basis functions to map the operator's input space.
       Interesting to use when the outputs' behaviors (e.g. frequency) are similar.
    """
    def forward(self, model, xb, xt):
        trunk_out = model.get_trunk_output(0, xt)
        outputs = []
        for i in range(model.n_outputs):
            branch_out = model.get_branch_output(i, xb)
            output = torch.matmul(trunk_out, branch_out).T
            outputs.append(output)
        return tuple(outputs)