from abc import ABC, abstractmethod
import torch
from ..factories.network_factory import NetworkFactory
from ...utilities.log_functions import pprint_layer_dict
from ..factories.component_factory import branch_factory, trunk_factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet
    from ..components import BaseBranch, BaseTrunk


class OutputHandling(ABC):
    def __init__(self) -> None:
        self.branch_output_size = None
        self.trunk_output_size = None
        self.n_trunk_outputs = None
        self.n_branch_outputs = None

    @property
    @abstractmethod
    def BASIS_CONFIG(self):
        pass
        
    @abstractmethod
    def forward(self, model: 'DeepONet', xb: torch.Tensor | None=None, xt: torch.Tensor | None=None) -> None:
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

    def create_components(self, model: "DeepONet", branch_config: dict, trunk_config: dict, branch_output_size: int, trunk_output_size: int, **kwargs) -> tuple["BaseBranch", "BaseTrunk"]:
        if hasattr(model, "pod_basis") and model.pod_basis is not None: # NOT WORKING YEt
            pod_helper = kwargs.get("pod_helper", None)
            if pod_helper is not None:
                n_modes, basis, mean = pod_helper.compute_modes(model, self.BASIS_CONFIG)
            model.n_basis_functions = n_modes
            trunk_config["type"] = "data"
            trunk_config["data"] = basis, mean
        else:
            trunk_config["type"] = trunk_config.get("type", "trainable")
        
        branch_config = branch_config.copy()
        branch_config['layers'].append(branch_output_size)
        branch_config_for_module = branch_config.copy()
        branch_config_for_module.pop("type", None)
        branch_config["module"] = NetworkFactory.create_network(branch_config_for_module)

        trunk_config = trunk_config.copy()
        trunk_config['layers'].append(trunk_output_size)
        trunk_config_for_module = trunk_config.copy()
        trunk_config_for_module.pop("type", None)
        trunk_config["module"] = NetworkFactory.create_network(trunk_config_for_module)

        trunk = trunk_factory(trunk_config)
        branch = branch_factory(branch_config)
        return branch, trunk
