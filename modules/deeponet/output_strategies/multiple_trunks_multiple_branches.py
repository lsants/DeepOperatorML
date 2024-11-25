import torch
from .output_handling_base import OutputHandlingStrategy
    
class MultipleTrunksMultipleBranchesStrategy(OutputHandlingStrategy):
    """For outputs that vary significantly between basis mapping as well as function inputs.
    """
    def forward(self, model, xb, xt):
        outputs = []
        for i in range(model.n_outputs):
            branch_out = model.get_branch_output(i, xb)
            trunk_out = model.get_trunk_output(i, xt)
            output = torch.matmul(trunk_out, branch_out).T
            outputs.append(output)
        return tuple(outputs)

