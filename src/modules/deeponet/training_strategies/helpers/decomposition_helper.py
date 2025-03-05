import torch
import logging
from ..training_strategy_base import TrainingStrategy
from typing import Any, Dict, Tuple, Optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class DecompositionHelper:
    def __init__(self) -> None:
        self.Q: Optional[torch.Tensor] = None
        self.R: Optional[torch.Tensor] = None
        self.T: Optional[torch.Tensor] = None
        self.trained_trunk: Optional[torch.Tensor] = None

    def decompose(self, model: 'DeepONet', params: Dict[str, Any], trunk_input: torch.Tensor) -> None:
        """Performs QR or SVD decomposition on the trunk output and updates matrices."""
        with torch.no_grad():
            decomposition = params.get('TRUNK_DECOMPOSITION', 'qr')
            phi = model.trunk.forward(trunk_input)
            if decomposition.lower() == 'qr':
                logger.info("Performing QR decomposition on trunk output...")
                Q, R = torch.linalg.qr(phi)
            elif decomposition.lower() == 'svd':
                logger.info("Performing SVD on trunk output...")
                Q, S, V = torch.linalg.svd(phi, full_matrices=False)
                R = torch.diag(S) @ V
            else:
                raise ValueError(f"Unknown decomposition method: {decomposition}")
            self.Q = Q
            self.R = R
            self.T = torch.linalg.inv(R)
            self.trained_trunk = self.Q @ self.R @ self.T
            logger.info(f"Decomposition complete: Q {Q.shape}, R {R.shape}, T {self.T.shape}")
            logger.info(f"Trained trunk is set with shape: {self.trained_trunk.shape}")
        return self.trained_trunk
