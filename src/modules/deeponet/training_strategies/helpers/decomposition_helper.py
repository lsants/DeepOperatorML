import torch
import logging
from ...deeponet import DeepONet
from typing import Any, Dict, Tuple, Optional
from training_strategy_base import TrainingStrategy

logger = logging.getLogger(__name__)

class DecompositionHelper:
    def __init__(self) -> None:
        self.Q: Optional[torch.Tensor] = None
        self.R: Optional[torch.Tensor] = None
        self.T: Optional[torch.Tensor] = None
        self.trained_trunk: Optional[torch.Tensor] = None

    def perform_decomposition(self, model: DeepONet, params: Dict[str, Any], xt: torch.Tensor) -> None:
        """Performs QR or SVD decomposition on the trunk output and updates matrices."""
        with torch.no_grad():
            decomposition = params.get('TRUNK_DECOMPOSITION', 'qr')
            phi = model.trunk.forward(xt)
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
