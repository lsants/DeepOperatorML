import torch
from abc import ABC, abstractmethod

class OutputHandlingStrategy(ABC):
    @abstractmethod
    def forward(self, model, xb, xt):
        """Defines how outputs are handled during the model's forward pass.

        Args:
            model (DeepONet): Model instance.
            xb (torch.Tensor): Input to the branch network.
            xt (torch.Tensor): Input to the trunk network.
        Returns:
            tuple: outputs as determined by the strategy.
        """
        pass

class SingleTrunkSplitBranchStrategy(OutputHandlingStrategy):
    """Use a single set of basis functions and one single input function mapping
       is to learn all outputs.
    """
    def forward(self, model, xb, xt):
        outputs = []
        trunk_out = model.get_trunk_output(0, xt)
        num_basis = trunk_out.shape[-1]
        for i in range(model.n_outputs):
            print(model.get_branch_output(0, xb).shape, 'AAAAAAAAAAAAaaa', i)
            branch_out = model.get_branch_output(0, xb)[i * num_basis : (i + 1) * num_basis , : ]
            print(trunk_out.shape, branch_out.shape, 'AAAAAAAAAAAAAAAAAa', i)
            print(i * num_basis , (i + 1) * num_basis, i )
            output = torch.matmul(trunk_out, branch_out).T
            outputs.append(output)
        return tuple(outputs)
    
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
    
class MultipleTrunksMultipleBranchesStrategy(OutputHandlingStrategy):
    """For outputs that vary significa
    """
    def forward(self, model, xb, xt):
        outputs = []
        for i in range(model.n_outputs):
            branch_out = model.get_branch_output(i, xb)
            trunk_out = model.get_trunk_output(i, xt)
            output = torch.matmul(trunk_out, branch_out).T
            outputs.append(output)
        return tuple(outputs)

