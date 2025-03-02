# modules/model/deeponet.py
import torch
import logging
from .nn.mlp import MLP
from .nn.kan import ChebyshevKAN
from .nn.resnet import ResNet

NETWORK_ARCHITECTURES = {
    'mlp': MLP,
    'kan': ChebyshevKAN,
    'resnet': ResNet
}

logger = logging.getLogger(__name__)

class DeepONet(torch.nn.Module):
    def __init__(self, branch_config, trunk_config, output_strategy, training_strategy, n_outputs, n_basis_functions, **kwargs):
        """Initializes the DeepONet model with specified strategies.

        Args:
            branch_config (dict): Configuration of the branch networks.
            trunk_config (dict): Configuration of the trunk networks.
            output_strategy (OutputHandlingStrategy): Strategy for handling the outputs.
            training_strategy (TrainingStrategy): Strategy for training.
            n_outputs (int): Number of outputs.
            data (optional): Data for strategies requiring pre-computed basis.
            var_share (optional): For POD.
        """
        super(DeepONet, self).__init__()
        self.n_outputs = n_outputs
        self.n_basis_functions = n_basis_functions
        self.output_strategy = output_strategy
        self.training_strategy = training_strategy

        basis_config = self.output_strategy.get_basis_config()

        if self.training_strategy.prepare_before_configure:
            self.training_strategy.prepare_training(self, basis_config=basis_config)

        self.branch_network, self.trunk_network = self.output_strategy.configure_networks(
            self, 
            branch_config,
            trunk_config,
            **kwargs
        )

        if not self.training_strategy.prepare_before_configure:
            self.training_strategy.prepare_training(self, basis_config=basis_config)
        
        if hasattr(self.training_strategy, 'after_networks_configured'):
            self.training_strategy.after_networks_configured(self)

    def create_network(self, config):
        """Creates a neural network based on provided configuration.

        Args:
            config (dict): 
                - 'architecture': Name of the architecture (e.g., MLP), (str).
                - 'layers': List defining the number of neurons in each layer.
                - 'activation': Activation function name (str).
                - Additional architecture-specific parameters.

        Returns:
            nn.Module: Initialized network
        """
        config = config.copy()
        architecture_name = config.pop('architecture').lower()
        try:
            constructor = NETWORK_ARCHITECTURES[architecture_name]
        except KeyError:
            raise ValueError(f"Architecture '{architecture_name}' not implemented.")
        
        required_params = constructor.get_required_params() if hasattr(constructor, 'get_required_params') else []
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter '{param}' for architecture '{architecture_name}'.")

        return constructor(**config)

    def forward(self, xb=None, xt=None):
        """Forward pass that delegates to the training strategy's forward method.

        Args:
            xb (torch.Tensor): Inputs to branch networks.
            xt (torch.Tensor): Inputs to trunk networks.

        Returns:
            tuple: Outputs as determined by the output handling strategy.
        """
        return self.training_strategy.forward(self, xb, xt)
    
    def get_trunk_output(self, xt):
        """
        Retrieves the trunk output. Delegates to TrainingStrategy.

        Args:
            xt (torch.Tensor): Input to the trunk network.

        Returns:
            torch.Tensor: Trunk output.
        """
        return self.training_strategy.get_trunk_output(self, xt)
    
    def get_branch_output(self, xb):
        """
        Retrieves the branch output. Delegates to TrainingStrategy.

        Args:
            xb (torch.Tensor): Input to the branch network.

        Returns:
            torch.Tensor: Branch output.
        """
        return self.training_strategy.get_branch_output(self, xb)
    
    def set_training_phase(self, phase):
        """
        Updates the training phase using the training strategy.

        Args:
            phase (str): The new training phase.
        """
        self.training_strategy.update_training_phase(self, phase)
