import torch 
from .output_handling_base import OutputHandlingStrategy

class MultipleTrunksSplitBranchStrategy(OutputHandlingStrategy):
    """Using multiple trunk networks to map the operator's input space.
       This allows learning of significantly different basis functions for each outputs.
       Here, it is considered that the input function mapping is learnable with a single network.
    """
    def forward(self, model, xb, xt):
        outputs = []
        for i in range(model.n_outputs):
            trunk_out = model.get_trunk_output(i, xt)
            num_basis = trunk_out.shape[-1]
            branch_out = model.get_branch_output(0, xb)[i * num_basis : (i + 1) * num_basis , : ]
            output = torch.matmul(trunk_out, branch_out).T
            outputs.append(output)
        return tuple(outputs)