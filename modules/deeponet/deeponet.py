# modules/model/deeponet.py

import torch
from .nn.mlp import MLP
from .nn.kan import ChebyshevKAN
from .nn.resnet import ResNet

NETWORK_ARCHITECTURES = {
    'mlp': MLP,
    'kan': ChebyshevKAN,
    'resnet': ResNet
}


class DeepONet(torch.nn.Module):
    def __init__(self, branch_config, trunk_config, output_strategy, training_strategy, n_outputs, **kwargs):
        """Initializes the DeepONet model with specified strategies.

        Args:
            branch_config (dict): Configuration of the branch networks.
            trunk_config (dict): Configuration of the trunk networks.
            output_strategy (OutputHandlingStrategy): Strategy for handling the outputs.
            training_strategy (TrainingStrategy): Strategy for training.
            n_outputs (int): Number of outputs.
        """
        super(DeepONet, self).__init__()
        self.n_outputs = n_outputs
        self.output_strategy = output_strategy
        self.training_strategy = training_strategy

        self.branch_networks, self.trunk_networks = self.output_strategy.configure_networks(
            self, 
            branch_config, 
            trunk_config, 
            **kwargs
        )

        self.training_strategy.prepare_training(self)

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

    def forward(self, xb, xt):
        """Forward pass that delegates to the training strategy's forward method.

        Args:
            xb (torch.Tensor): Inputs to branch networks.
            xt (torch.Tensor): Inputs to trunk networks.

        Returns:
            tuple: Outputs as determined by the output handling strategy.
        """
        return self.training_strategy.forward(self, xb, xt)
    
    def get_trunk_output(self, i, xt_i):
        """
        Retrieves the trunk output for the i-th output. Delegates to TrainingStrategy.

        Args:
            i (int): Index of the output.
            xt_i (torch.Tensor): Input to the trunk network for the i-th output.

        Returns:
            torch.Tensor: Trunk output for the i-th output.
        """
        return self.training_strategy.get_trunk_output(self, i, xt_i)
    
    def get_branch_output(self, i, xb_i):
        """
        Retrieves the branch output for the i-th output. Delegates to TrainingStrategy.

        Args:
            i (int): Index of the output.
            xb_i (torch.Tensor): Input to the branch network for the i-th output.

        Returns:
            torch.Tensor: Branch output for the i-th output.
        """
        return self.training_strategy.get_branch_output(self, i, xb_i)
    
    def set_training_phase(self, phase):
        """
        Updates the training phase using the training strategy.

        Args:
            phase (str): The new training phase.
        """
        self.training_strategy.update_training_phase(self, phase)
