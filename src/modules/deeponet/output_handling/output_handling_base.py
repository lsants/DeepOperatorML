from abc import ABC, abstractmethod
import torch
from ..training_strategies import PODTrainingStrategy
from ..factories.network_factory import NetworkFactory
from ..factories.component_factory import branch_factory, trunk_factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet
    from ..components import BaseBranch, BaseTrunk


class OutputHandling(ABC):
    def __init__(self) -> None:
        self.branch_output_size = None
        self.trunk_output_size = None

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

    def config_basis(self,  model: "DeepONet", trunk_config: dict):
        if isinstance(model.training_strategy, PODTrainingStrategy):
            trunk_config["type"] = "data"
            if not model.training_strategy.inference:
                if hasattr(model.training_strategy, 'pod_helper'): # this should always be True
                    n_modes, basis, mean = model.training_strategy.pod_helper.compute_modes(model, self.BASIS_CONFIG)
                else:
                    raise ValueError("POD Helper not initialized for training.")
                model.n_basis_functions = n_modes
                trunk_config["data"] = {'basis': basis, 'mean': mean}
            else:
                trunk_config["data"] = model.training_strategy.pod_trunk
                if self.BASIS_CONFIG == 'single':
                    model.n_basis_functions = trunk_config["data"]["basis"].shape[-1]
                else:
                    model.n_basis_functions = trunk_config["data"]["basis"].shape[-1] // model.n_outputs
                if not trunk_config["data"]:
                    raise ValueError("Error! POD trunk not found.")
        else:
            trunk_config["type"] = trunk_config.get("type", "trainable")
        return trunk_config

    def create_components(self, model: "DeepONet", branch_config: dict, trunk_config: dict, branch_output_size: int, trunk_output_size: int, **kwargs) -> tuple["BaseBranch", "BaseTrunk"]:
        branch_config = branch_config.copy()
        branch_config['layers'].append(branch_output_size)
        branch_config_for_module = branch_config.copy()
        branch_config_for_module.pop("type", None)
        if branch_config["type"] == 'trainable':
            branch_config["module"] = NetworkFactory.create_network(branch_config_for_module)

        trunk_config = trunk_config.copy()
        trunk_config['layers'].append(trunk_output_size)
        trunk_config_for_module = trunk_config.copy()
        trunk_config_for_module.pop("type", None)
        if trunk_config["type"] == 'trainable':
            trunk_config["module"] = NetworkFactory.create_network(trunk_config_for_module)

        trunk = trunk_factory(trunk_config)
        branch = branch_factory(branch_config)
        return branch, trunk
