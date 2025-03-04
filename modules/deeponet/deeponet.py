# modules/model/deeponet.py
import torch
import logging
from .training_strategies import TrainingStrategy
from ..factories.network_factory import NetworkFactory
from .output_strategies.output_handling_base import OutputHandlingStrategy

logger = logging.getLogger(__name__)

class DeepONet(torch.nn.Module):
    def __init__(self, branch_config: dict, trunk_config: dict, output_strategy: OutputHandlingStrategy, training_strategy: TrainingStrategy, n_outputs: int, n_basis_functions: int, **kwargs) -> None:
        """Initializes the DeepONet model with specified strategies.

        Args:
            branch_config (dict): Configuration of the branch networks.
            trunk_config (dict): Configuration of the trunk networks.
            output_strategy (OutputHandlingStrategy): Strategy for handling the outputs.
            training_strategy (TrainingStrategy): Strategy for training.
            n_outputs (int): Number of outputs.
            n_basis_functions (int): Number of basis functions.

        """
        super(DeepONet, self).__init__()
        self.n_outputs: int = n_outputs
        self.n_basis_functions: int = n_basis_functions
        self.output_strategy: OutputHandlingStrategy = output_strategy
        self.training_strategy: TrainingStrategy = training_strategy

        if self.training_strategy.prepare_before_configure:
            self.training_strategy.prepare_training(self)

        self.branch_network, self.trunk_network = self.output_strategy.configure_networks(
            self, 
            branch_config,
            trunk_config,
            **kwargs
        )

        if not self.training_strategy.prepare_before_configure:
            self.training_strategy.prepare_training(self)
        
        if hasattr(self.training_strategy, 'after_networks_configured'):
            self.training_strategy.after_networks_configured(self)

    def create_network(self, config: dict) -> torch.nn.Module:
        """Creates a neural network based on provided configuration.

        Args:
            config (dict): Configuration of the network.

        Returns:
            torch.nn.Module: Initialized network
        """
        return NetworkFactory.create_network(config)

    def forward(self, xb: torch.Tensor | None=None, xt: torch.Tensor | None=None) -> tuple[torch.Tensor]:
        """Forward pass that delegates to the training strategy's forward method.

        Args:
            xb (torch.Tensor): Inputs to branch networks.
            xt (torch.Tensor): Inputs to trunk networks.

        Returns:
            tuple: Outputs as determined by the output handling strategy.
        """
        return self.training_strategy.forward(self, xb, xt)
    
    def get_trunk_output(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the trunk output. Delegates to TrainingStrategy.

        Args:
            xt (torch.Tensor): Input to the trunk network.

        Returns:
            torch.Tensor: Trunk output.
        """
        return self.training_strategy.get_trunk_output(self, xt)
    
    def get_branch_output(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the branch output. Delegates to TrainingStrategy.

        Args:
            xb (torch.Tensor): Input to the branch network.

        Returns:
            torch.Tensor: Branch output.
        """
        return self.training_strategy.get_branch_output(self, xb)
    
    def set_training_phase(self, phase: str) -> None:
        """
        Updates the training phase using the training strategy.

        Args:
            phase (str): The new training phase.
        """
        self.training_strategy.update_training_phase(self, phase)
