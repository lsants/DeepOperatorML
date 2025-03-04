
import torch
import logging
from .training_strategy_base import TrainingStrategy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class PODTrainingStrategy(TrainingStrategy):
    def __init__(self, loss_fn: callable, inference: bool, data: torch.Tensor | None=None, var_share: float | None=None) -> None:
        super().__init__(loss_fn)
        self.data = data
        self.var_share = var_share
        self.inference = inference
        self.pod_basis = None
        self.mean_functions = None
        self.prepare_before_configure = True

    def prepare_training(self, model: 'DeepONet', **kwargs) -> None:
        """
        Prepares the model for POD-based training by integrating POD basis and freezing the trunk networks.

        Args:
            model (DeepONet): The model instance.
        """
        if not self.inference:
            basis_config = model.output_strategy.BASIS_CONFIG
            if basis_config == 'single':
                n_basis = self._prepare_single_basis(model)
                model.n_basis_functions = n_basis
            elif basis_config == 'multiple':
                n_basis = self._prepare_multiple_basis(model)
                model.n_basis_functions = n_basis
            else:
                raise ValueError(
                    f"Unknown 'basis_config' type: {basis_config}")
            
    def set_pod_data(self, pod_basis: torch.Tensor, mean_functions: torch.Tensor) -> None:
        self.pod_basis = pod_basis
        self.mean_functions = mean_functions
    
    def _prepare_single_basis(self, model: 'DeepONet', **kwargs) -> int:
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

        n_modes = most_significant_modes
        basis = U[ : , : n_modes]
        pod_basis_list.append(basis)
        
        self.pod_basis = basis
        self.mean_functions = mean.unsqueeze(0)

        logger.info(f"\n Using {n_modes} modes for {variance_share:.2%} of variance.\n")
        logger.info(f"\n Basis functions, mean functions: {self.pod_basis.shape}, {self.mean_functions.shape}.\n")
        model.register_buffer(f'pod_basis', self.pod_basis)
        model.register_buffer(f'mean_functions', self.mean_functions)

        return n_modes

    def _prepare_multiple_basis(self, model: 'DeepONet', **kwargs) -> int:
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

        modes_for_each_output = []
        for output_name in outputs_names:
            output = self.data[output_name]

            mean = torch.mean(output, dim=0)
            centered = (output - mean).T

            U, S, _ = torch.linalg.svd(centered)
            explained_variance_ratio = torch.cumsum(
                S ** 2, dim=0) / torch.linalg.norm(S) ** 2
            if variance_share:
                most_significant_modes = (
                    explained_variance_ratio < variance_share).sum() + 1
            else:
                raise ValueError("Variance share was not given. \nThere's no way to know how many modes should be used.")

            n_modes = most_significant_modes
            modes_for_each_output.append(n_modes)
            logger.debug(f"OUTPUT SHAPE, {output.shape}")
            logger.debug(f"U SHAPE, {U.shape}")
            logger.debug(f"S SHAPE, {S.shape}")
            logger.debug(f"NUM MODES, {n_modes}")

        n_modes = max(modes_for_each_output)
        for output_name in outputs_names:
            output = self.data[output_name]

            mean = torch.mean(output, dim=0)
            mean_functions_list.append(mean)
            centered = (output - mean).T

            U, S, _ = torch.linalg.svd(centered)
            basis = U[ : , : n_modes]
            pod_basis_list.append(basis)

        self.pod_basis = torch.concatenate(pod_basis_list, dim=-1)
        self.mean_functions = torch.stack(mean_functions_list, dim=0)

        logger.info(f"\n Using {n_modes} modes for {variance_share:.2%} of variance.\n")
        logger.info(f"\n Basis functions, mean functions: {self.pod_basis.shape}, {self.mean_functions.shape}.\n")
        model.register_buffer(f'pod_basis', self.pod_basis)
        model.register_buffer(f'mean_functions', self.mean_functions)

        return n_modes

    def get_basis_functions(self, **kwargs) -> torch.Tensor:
        trunk_output = self.pod_basis
        if trunk_output.ndim == 2:
            trunk_output = trunk_output.unsqueeze(-1)
        basis_functions = torch.transpose(trunk_output, 0, 1)
        return basis_functions

    def get_trunk_output(self, model: 'DeepONet', xt: torch.Tensor) -> torch.Tensor:
        """
        Overrides the trunk output to use pod_basis instead of trunk_network.

        Args:
            model (DeepONet): The model instance.
            xt (torch.Tensor): Input to the trunk network (unused).

        Returns:
            torch.Tensor: Trunk output, which is pod_basis.
        """

        return self.pod_basis

    def get_branch_output(self, model: 'DeepONet', xb: torch.Tensor) -> torch.Tensor:
        """
        Optionally, modify the branch output if needed. For POD, branches are used as is.

        Args:
            model (DeepONet): The model instance.
            xb (torch.Tensor): Input to the branch network.

        Returns:
            torch.Tensor: Branch output.
        """

        return model.branch_network(xb).T

    def forward(self, model: 'DeepONet', xb: torch.Tensor | None = None, xt: torch.Tensor | None = None) -> tuple[torch.Tensor]:
        pod_basis = self.pod_basis
        dot_product = model.output_strategy.forward(model, 
                                             data_branch=xb, 
                                             data_trunk=None,
                                             matrix_branch=None,
                                             matrix_trunk=pod_basis)
        m = self.mean_functions.shape[0]
        if m == 1:
            output = tuple(x + self.mean_functions for x in dot_product)
        else:
            if len(dot_product) != m:
                raise ValueError("When m > 1, the number of outputs must be equal to m.")
            output = tuple(dot_product[i] + self.mean_functions[i : i + 1] for i in range(m))
        return output

    def after_epoch(self, epoch: int, model:'DeepONet', params: dict, **kwargs) -> None:
        if epoch > 1:
            return
        params['BASIS_FUNCTIONS'] = model.n_basis_functions
