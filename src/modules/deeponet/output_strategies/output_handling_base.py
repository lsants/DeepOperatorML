from abc import ABC, abstractmethod
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

## Intuition: # of networks determines which one is in the loop. size determines slicing

class OutputHandlingStrategy(ABC):
    def __init__(self):
        self.branch_output_size = None
        self.trunk_output_size = None
        self.n_trunk_outputs = None
        self.n_branch_outputs = None

    @property
    @abstractmethod
    def BASIS_CONFIG(self):
        pass
        
    @abstractmethod
    def forward(self, model: 'DeepONet', xb: torch.Tensor | None=None, xt: torch.Tensor | None=None):
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
    def configure_networks(self, model: 'DeepONet', branch_config: dict, trunk_config: dict, **kwargs):
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