import logging
import torch
from .output_handling_base import OutputHandling
from ..factories.component_factory import branch_factory, trunk_factory
from ...utilities.log_functions import pprint_layer_dict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class SingleOutputHandling(OutputHandling):
    """Use a single set of basis functions and one single input function mapping
       to learn one single output.
    """
    def __init__(self):
        super().__init__()
        
    @property
    def BASIS_CONFIG(self):
        return 'single'
    
    def configure_components(self, model, branch_config: dict, trunk_config: dict, **kwargs) -> tuple:
        if hasattr(model, "pod_basis") and model.pod_basis is not None:
            pod_helper = kwargs.get("pod_helper", None)
            if pod_helper is not None:
                n_modes = pod_helper.compute_modes(model)
            model.n_basis_functions = n_modes
            trunk_config["type"] = "fixed"
            trunk_config["fixed_tensor"] = model.pod_basis
        else:
            trunk_config["type"] = trunk_config.get("type", "trainable")

        n_basis_functions = model.n_basis_functions
        trunk_output_size = n_basis_functions
        branch_output_size = n_basis_functions

        trunk_config = trunk_config.copy()
        trunk_config['layers'].append(trunk_output_size)

        branch_config = branch_config.copy()
        branch_config['layers'].append(branch_output_size)

        trunk = model.trunk_factory(trunk_config)
        branch = model.branch_factory(branch_config)

        logger.debug(f"SingleOutputHandling: Computed trunk output size: {trunk_output_size}")
        logger.debug(f"SingleOutputHandling: Computed branch output size: {branch_output_size}")
        logger.debug(f"SingleOutputHandling: Trunk layers: {pprint_layer_dict(trunk_config.get('layers', []))}")
        logger.debug(f"SingleOutputHandling: Branch layers: {pprint_layer_dict(branch_config.get('layers', []))}")

        return branch, trunk
    
def forward(self, model: DeepONet, trunk_out: torch.Tensor, branch_out: torch.Tensor) -> tuple[torch.Tensor]: 
    """
    For SingleOutputHandling, a single aggregated output is produced by directly multiplying
    the trunk and branch outputs.
    """
    output = torch.matmul(trunk_out, branch_out).T
    return (output,)


