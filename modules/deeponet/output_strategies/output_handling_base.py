from abc import ABC, abstractmethod

## Intuition: # of networks determines which one is in the loop. size determines slicing

class OutputHandlingStrategy(ABC):
    def __init__(self):
        self.branch_output_size = None
        self.trunk_output_size = None
        self.n_trunk_outputs = None
        self.n_branch_outputs = None
        
    @abstractmethod
    def forward(self, model, xb=None, xt=None):
        """Defines how outputs are handled during the model's forward pass.

        Args:
            model (DeepONet): Model instance.
            xb (torch.Tensor): Input to the branch network.
            xt (torch.Tensor): Input to the trunk network.
        Returns:
            tuple: outputs as determined by the strategy.
        """
        pass

    @abstractmethod
    def configure_networks(self, model, branch_config, trunk_config, **kwargs):
        """
        Configures the number and output sizes of the networks (branch and trunk).

        Args:
            model (DeepONet): The model instance.
            branch_config (dict): Initial branch configuration.
            trunk_config (dict): Initial trunk configuration.

        Returns:
            tuple: (list of branch networks, list of trunk networks)
        """
        pass

    @abstractmethod
    def get_basis_config(self):
        pass