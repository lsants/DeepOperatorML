import logging
import torch
from .output_handling_base import OutputHandling
from ...utilities.log_functions import pprint_layer_dict
from ..factories.component_factory import trunk_factory, branch_factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class ShareBranchHandling(OutputHandling):
    def __init__(self):
        super().__init__()

    @property
    def BASIS_CONFIG(self):
        return 'multiple'

    def configure_components(self, model, branch_config: dict, trunk_config: dict, **kwargs) -> tuple:
        if hasattr(model, "pod_basis") and model.pod_basis is not None:
            pod_helper = kwargs.get("pod_helper", None)
            if pod_helper is not None:
                n_modes = pod_helper.compute_modes(model, self.BASIS_CONFIG)
            model.n_basis_functions = n_modes
            trunk_config["type"] = "fixed"
            trunk_config["fixed_tensor"] = model.pod_basis
        else:
            trunk_config["type"] = trunk_config.get("type", "trainable")

        n_basis_functions = model.n_basis_functions
        trunk_output_size = n_basis_functions * model.n_outputs
        branch_output_size = n_basis_functions

        trunk_config = trunk_config.copy()
        trunk_config['layers'].append(trunk_output_size)

        branch_config = branch_config.copy()
        branch_config['layers'].append(branch_output_size)

        trunk = model.trunk_factory(trunk_config)
        branch = model.branch_factory(branch_config)

        logger.debug(f"ShareBranchHandling: Computed trunk output size: {trunk_output_size}")
        logger.debug(f"ShareBranchHandling: Computed branch output size: {branch_output_size}")
        logger.debug(f"ShareBranchHandling: Trunk layers: {pprint_layer_dict(trunk_config.get('layers', []))}")
        logger.debug(f"ShareBranchHandling: Branch layers: {pprint_layer_dict(branch_config.get('layers', []))}")

        return branch, trunk
    
def forward(self, model: DeepONet, trunk_out: torch.Tensor, branch_out: torch.Tensor) -> tuple[torch.Tensor]: 
    """
    For ShareBranchHandling, the branch output remains un-split, while the trunk output
    is split into segments of size equal to model.n_basis_functions (one segment per output).
    The final output for each operator is computed by multiplying the corresponding trunk slice 
    with the branch output.
    """
    N = model.n_basis_functions
    outputs = []
    for i in range(model.n_outputs):
        trunk_slice = trunk_out[:, i * N:(i + 1) * N]
        output = torch.matmul(trunk_slice, branch_out).T
        outputs.append(output)
    return tuple(outputs)