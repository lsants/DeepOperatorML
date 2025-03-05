# File: src/modules/deeponet/helpers/pod_basis_helper.py
import torch
import logging
from typing import TYPE_CHECKING, Any, Dict, Tuple, Optional
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class PODBasisHelper:
    def __init__(self, data: torch.Tensor | dict | None = None, var_share: float | None = None):
        """
        Initializes the PODBasisHelper.
        
        Args:
            data: Training data tensor or a dictionary of tensors (if outputs are separated).
            var_share: The variance share threshold (e.g., 0.9 for 90% of variance).
        """
        self.data = data
        self.var_share = var_share
        self.pod_basis = None
        self.mean_functions = None

    def set_pod_data(self, pod_basis: torch.Tensor, mean_functions: torch.Tensor) -> None:
        """
        Explicitly sets the POD basis and mean functions.
        """
        self.pod_basis = pod_basis
        self.mean_functions = mean_functions

    def get_pod_data(self) -> tuple[torch.Tensor]:
        return self.pod_basis, self.mean_functions

    def compute_modes(self, model: "DeepONet", basis_config: str, **kwargs) -> int:
        """
        Computes the POD modes based on the provided BASIS_CONFIG.
        
        Args:
            model: The DeepONet model instance.
            basis_config: A string indicating the basis type ("single" or "multiple").
            **kwargs: Additional arguments (for example, training data).
        
        Returns:
            n_modes (int): The number of modes selected.
            
        Also updates:
            - model.n_basis_functions to reflect the number of modes.
            - Registers 'pod_basis' and 'mean_functions' as buffers in model.
        """
        if basis_config == "single":
            n_modes = self._prepare_single_basis(model, **kwargs)
        elif basis_config == "multiple":
            n_modes = self._prepare_multiple_basis(model, **kwargs)
        else:
            raise ValueError(f"Unknown basis_config '{basis_config}'. Expected 'single' or 'multiple'.")
        return n_modes

    def _prepare_single_basis(self, model: "DeepONet", **kwargs) -> int:
        """
        Computes a single set of POD basis functions for all outputs.
        Concatenates all output data from self.data, computes mean and SVD,
        and selects enough modes to capture the desired variance share.
        
        Returns:
            n_modes (int): The number of modes selected.
        """
        if self.data is None:
            raise ValueError("Data must be provided for POD basis computation.")

        variance_share = self.var_share
        output_keys = [k for k in self.data.keys() if k not in ['xb', 'xt', 'index']]
        
        outputs = torch.cat([self.data[k] for k in output_keys], dim=0)
        mean = torch.mean(outputs, dim=0, keepdim=True)
        centered = (outputs - mean).T
        
        U, S, _ = torch.linalg.svd(centered, full_matrices=False)
        explained_variance_ratio = torch.cumsum(S**2, dim=0) / torch.linalg.norm(S)**2
        if variance_share is None:
            raise ValueError("Variance share must be provided for POD basis computation.")
        n_modes = (explained_variance_ratio < variance_share).sum().item() + 1
        
        basis = U[:, :n_modes]  # shape (features, n_modes)
        self.pod_basis = basis
        self.mean_functions = mean  # shape (1, features)
        
        logger.info(f"PODBasisHelper (single): Using {n_modes} modes for {variance_share*100:.2f}% variance.")
        logger.info(f"PODBasisHelper (single): Basis shape: {self.pod_basis.shape}, Mean shape: {self.mean_functions.shape}")
        
        model.register_buffer("pod_basis", self.pod_basis)
        model.register_buffer("mean_functions", self.mean_functions)
        model.n_basis_functions = n_modes
        return n_modes

    def _prepare_multiple_basis(self, model: "DeepONet", **kwargs) -> int:
        """
        Computes multiple sets of POD basis functions, one per output.
        For each output key in self.data (except reserved keys), compute the basis
        and then select the maximum number of modes needed across outputs.
        
        Returns:
            n_modes (int): The maximum number of modes across outputs.
        """
        if self.data is None:
            raise ValueError("Data must be provided for POD basis computation.")

        output_keys = [k for k in self.data.keys() if k not in ['xb', 'xt', 'index']]
        modes_per_output = []
        pod_basis_list = []
        mean_functions_list = []
        
        for key in output_keys:
            output = self.data[key]
            mean = torch.mean(output, dim=0, keepdim=True)  # shape (1, features)
            centered = (output - mean).T
            U, S, _ = torch.linalg.svd(centered, full_matrices=False)
            explained_variance_ratio = torch.cumsum(S**2, dim=0) / torch.linalg.norm(S)**2
            n_modes = (explained_variance_ratio < self.var_share).sum().item() + 1
            modes_per_output.append(n_modes)
            pod_basis_list.append(U[:, :n_modes])
            mean_functions_list.append(mean)
        
        n_modes = max(modes_per_output)
        truncated_bases = [basis[:, : n_modes] for basis in pod_basis_list]
        self.pod_basis = torch.cat(truncated_bases, dim=-1)  # shape: (features, n_modes * num_outputs)
        self.mean_functions = torch.stack(mean_functions_list, dim=0)  # shape: (num_outputs, features)
        
        logger.info(f"PODBasisHelper (multiple): Using {n_modes} modes per output for {self.var_share*100:.2f}% variance.")
        logger.info(f"PODBasisHelper (multiple): Basis shape: {self.pod_basis.shape}, Mean shape: {self.mean_functions.shape}")
        
        model.register_buffer("pod_basis", self.pod_basis)
        model.register_buffer("mean_functions", self.mean_functions)
        model.n_basis_functions = n_modes
        return n_modes, self.pod_basis, self.mean_functions
