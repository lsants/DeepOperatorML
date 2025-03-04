from abc import ABC, abstractmethod
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

## Intuition: # of networks determines which one is in the loop. size determines slicing

class OutputHandling(ABC):
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
    def configure_components(self, model: "DeepONet", branch_component: object, trunk_component: object, branch_config: dict, trunk_config: dict, **kwargs) -> tuple:
        """
        Uses the provided branch and trunk configuration dictionaries (augmented by training strategy info)
        to create and adjust the networks. This method can also use additional data (e.g. POD basis)
        to update the configurations and even replace the trunk component if needed.
        
        Returns:
            tuple: (branch_component, trunk_component)
        """
        pass