import logging
import torch
from typing import TYPE_CHECKING
from src.modules.models.deeponet.components.output_handler.config import OutputConfig
from src.modules.models.deeponet.components.output_handler.registry import OutputRegistry
from src.modules.models.deeponet.components.output_handler.protocol import OutputHandler
if TYPE_CHECKING:
    from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig

logger = logging.getLogger(__name__)


@OutputRegistry.register("two_step_final")
class Phase2Handler(torch.nn.Module):
    """
    A simple, universal handler for combining standardized Phase 2 outputs.
    Crucially, it also adapts a DeepONetConfig for inference-time model loading.
    """
    def __init__(self, config: 'OutputConfig'): # Must accept a config object
        super().__init__()
        self.num_channels = config.num_channels

    def adjust_dimensions(self, config: 'DeepONetConfig'):
        """
        Reconfigures the DeepONetConfig to build a serializable Phase 2 model.
        This method is called by the factory BEFORE instantiating the model for inference.
        """
        return

    def combine(self, branch_out: torch.Tensor, trunk_out: torch.Tensor) -> torch.Tensor:
        """
        Performs a universal channel-wise dot product.
        - branch_out shape: (B, C, P)
        - trunk_out shape: (T, C, P) or (T, P)
        """
        if trunk_out.ndim == 2:
            # SharedTrunk 
            return torch.einsum('bcp,tp->btc', branch_out, trunk_out)
        else:
            # SharedBranch and SplitOutputs
            return torch.einsum('bcp,tcp->btc', branch_out, trunk_out)