import torch
from  .training_strategy_base import TrainingStrategy
from ..loss_functions.loss_complex import loss_complex

class PODTrainingStrategy(TrainingStrategy):
    def __init__(self, pod_basis, mean_functions):
        """
        Initializes the POD training strategy.

        Args:
            pod_basis (torch.Tensor): Precomputed POD basis matrices.
                                       Shape: (n_outputs, num_modes, features)
            mean_functions (torch.Tensor): Precomputed mean functions.
                                           Shape: (n_outputs, features)
        """
        self.pod_basis = pod_basis
        self.mean_functions = mean_functions

    def prepare_training(self, model):
        """
        Prepares the model for POD-based training by integrating POD basis and freezing the trunk networks.

        Args:
            model (DeepONet): The model instance.
        """
        for i in range(model.n_outputs):
            model.register_buffer(f'pod_basis_{i}', self.pod_basis[i])
            model.register_buffer(f'mean_functions_{i}', self.mean_functions[i])
        
        for trunk in model.trunk_networks:
            for param in trunk.parameters():
                param.requires_grad = False

    def update_training_phase(self, model, phase, params):
        """
        Not used here.

        Args:
            model (DeepONet): The model instance.
            phase (str): Not used in POD training.
        """
        pass

    def get_trunk_output(self, model, i, xt_i):
        """
        Overrides the trunk output to use pod_basis instead of trunk_network.

        Args:
            model (DeepONet): The model instance.
            i (int): Index of the output.
            xt_i (torch.Tensor): Input to the trunk network for the i-th output (unused).

        Returns:
            torch.Tensor: Trunk output for the i-th output, which is pod_basis.
        """
        return getattr(model, f'pod_basis_{i}')
    
    def get_branch_output(self, model, i, xb_i):
        """
        Optionally, modify the branch output if needed. For POD, branches are used as is.

        Args:
            model (DeepONet): The model instance.
            i (int): Index of the output.
            xb_i (torch.Tensor): Input to the branch network for the i-th output.

        Returns:
            torch.Tensor: Branch output for the i-th output.
        """
        return model.branch_networks[i](xb_i)
    
