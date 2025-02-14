import torch
import logging
from .training_strategy_base import TrainingStrategy
from ..optimization.loss_complex import loss_complex

class PODTrainingStrategy(TrainingStrategy):
    def __init__(self, data=None, var_share=None, inference=False):
        super().__init__()
        self.data = data
        self.var_share = var_share
        self.inference = inference
        self.pod_basis = None
        self.mean_functions = None
        self.prepare_before_configure = True

    def prepare_training(self, model, **kwargs):
        """
        Prepares the model for POD-based training by integrating POD basis and freezing the trunk networks.

        Args:
            model (DeepONet): The model instance.
        """

        if self.pod_basis is None:
            basis_config = kwargs.get("basis_config")

            if basis_config is None:
                raise ValueError(
                    "'basis_config' must be provided by 'OutputHandlingStrategy'")

            basis_type = basis_config.get('type')

            if not self.inference:
                if basis_type == 'single':
                    num_basis = self._prepare_single_basis(model)
                    model.n_basis_functions = num_basis
                elif basis_type == 'multiple':
                    num_basis = self._prepare_multiple_basis(model)
                    model.n_basis_functions = num_basis
                else:
                    raise ValueError(
                        f"Unknown 'basis_config' type: {basis_type}")

    def _prepare_single_basis(self, model, **kwargs):
        """Computes a single set of POD basis functions for all outputs.

        Args:
            model (DeepONet): Model instance.
        """

        if self.data is None:
            raise ValueError("Data must be provided for PODTrainingStrategy.")

        variance_share = self.var_share
        pod_basis_list = []
        mean_functions_list = []

        output_hashes = [k for k in self.data.keys() if k not in [
            'xb', 'xt', 'index']]

        # Reshape ouputs to shape (n_outputs * input_functions, coordinates)
        outputs = torch.Tensor()
        for output in output_hashes:
            outputs = torch.cat((outputs, self.data[output]), 0)

        mean = torch.mean(outputs, dim=0)
        mean_functions_list.append(mean)
        centered = (outputs - mean).T

        U, S, _ = torch.linalg.svd(centered)
        explained_variance_ratio = torch.cumsum(
            S**2, dim=0) / torch.linalg.norm(S)**2

        if variance_share:
            most_significant_modes = (
                explained_variance_ratio < variance_share).sum() + 1
        else:
            raise ValueError(
                "Variance share was not given. There's no way to know how many modes should be used.")

        num_modes = most_significant_modes
        basis = U[:, : num_modes]
        pod_basis_list.append(basis)

        self.pod_basis = torch.stack(pod_basis_list, dim=0)

        self.mean_functions = torch.stack(mean_functions_list, dim=0)

        model.register_buffer(f'pod_basis', self.pod_basis)
        model.register_buffer(f'mean_functions', self.mean_functions)

        return num_modes

    def _prepare_multiple_basis(self, model, **kwargs):
        """Computes multiple sets of POD basis functions, one for each output.

        Args:
            model (DeepONet): Model instance.
            data (torch.utils.data.Subset): Training data.
        """

        if self.data is None:
            raise ValueError("Data must be provided for PODTrainingStrategy.")

        output_keys = [i for i in self.data.keys() if i not in ['xb', 'xt', 'index']]
        outputs_names = output_keys
        variance_share = self.var_share
        pod_basis_list = []
        mean_functions_list = []

        for output_name in outputs_names:
            output = self.data[output_name]

            mean = torch.mean(output, dim=0)
            mean_functions_list.append(mean)
            centered = (output - mean).T

            U, S, _ = torch.linalg.svd(centered)
            explained_variance_ratio = torch.cumsum(
                S ** 2, dim=0) / torch.linalg.norm(S) ** 2
            if variance_share:
                most_significant_modes = (
                    explained_variance_ratio < variance_share).sum() + 1
            else:
                raise ValueError(
                    "Variance share was not given. There's no way to know how many modes should be used.")

            num_modes = most_significant_modes
            basis = U[:, : num_modes]
            pod_basis_list.append(basis)

        self.pod_basis = torch.stack(pod_basis_list, dim=0)
        self.mean_functions = torch.stack(mean_functions_list, dim=0)

        model.register_buffer(f'pod_basis', self.pod_basis)
        model.register_buffer(f'mean_functions', self.mean_functions)

    def get_basis_functions(self):
        return self.pod_basis

    def set_basis(self, pod_basis, mean_functions):
        self.pod_basis = pod_basis
        self.mean_functions = mean_functions

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

        pod_basis = self.pod_basis
        requested_basis = pod_basis[i]

        return requested_basis

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

        return model.branch_networks[i](xb_i).T

    def forward(self, model, xb=None, xt=None):
        pod_basis = self.pod_basis
        return model.output_strategy.forward(model, data_branch=xb, data_trunk=pod_basis)

    def after_epoch(self, epoch, model, params, **kwargs):
        if epoch > 1:
            return
        params['BASIS_FUNCTIONS'] = model.n_basis_functions
