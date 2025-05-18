import torch
import logging
from ..training_strategy_base import TrainingStrategy
from typing import Any, Dict, Tuple, Optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.model.deeponet import DeepONet

logger = logging.getLogger(__name__)

class DecompositionHelper:
    def __init__(self) -> None:
        self.Q: torch.Tensor | None = None
        self.R: torch.Tensor | None = None
        self.T: torch.Tensor | None = None
        self.trained_trunk: torch.Tensor | torch.Tensor = None

    def decompose(self, model: 'DeepONet', training_params: dict[str, any], trunk_input: torch.Tensor) -> None:
        """Performs QR or SVD decomposition on the trunk output and updates matrices."""
        with torch.no_grad():
            decomposition = training_params.get('TRUNK_DECOMPOSITION', 'svd')
            phi = model.trunk.forward(trunk_input)
            K = model.n_basis_functions
            n = model.n_outputs
            
            if phi.shape[1] == K:
                logger.info("Performing single-basis trunk decomposition...")
                if decomposition.lower() == 'qr':
                    logger.info("Performing QR decomposition on trunk output...")
                    Q, R = torch.linalg.qr(phi)
                elif decomposition.lower() == 'svd':
                    logger.info("Performing SVD on trunk output...")
                    Q, S, V = torch.linalg.svd(phi, full_matrices=False)
                    R = torch.diag(S) @ V
                else:
                    raise ValueError(f"Unknown decomposition method: {decomposition}")
                T = torch.linalg.inv(R)

                self.Q, self.R, self.T= Q, R, T
                self.trained_trunk = Q @ R @ T
            elif phi.shape[1] == n * K:
                logger.info("Performing split-output trunk decomposition...")
                Q_blocks, R_blocks, T_blocks = [], [], []

                for i in range(n):
                    phi_slice = phi[ : , i * K : (i + 1) * K]
                    if decomposition.lower() == 'qr':
                        logger.info(f"QR decomposition on output {i + 1}")
                        Q_i, R_i = torch.linalg.qr(phi_slice)
                    elif decomposition.lower() == 'svd':
                        logger.info(f"SVD on output {i + 1}")
                        Q_i, S_i, V_i = torch.linalg.svd(phi_slice, full_matrices=False)
                        R_i = torch.diag(S_i) @ V_i
                    else:
                        raise ValueError(f"Unknown decomposition method: {decomposition}")
                    
                    T_i = torch.linalg.inv(R_i)

                    Q_blocks.append(Q_i)
                    R_blocks.append(R_i)
                    T_blocks.append(T_i)

                self.Q = torch.cat(Q_blocks, dim=1)
                self.R = torch.block_diag(*R_blocks)
                self.T = torch.block_diag(*T_blocks)

                self.trained_trunk = self.Q @ self.R @ self.T
            else:
                raise ValueError(f"Unexpected phi shape {phi.shape}; check model output handling.")
            
            logger.info(f"Decomposition complete: Q {self.Q.shape}, R {self.R.shape}, T {self.T.shape}")
            logger.info(f"Trained trunk is set with shape: {self.trained_trunk.shape}")

        return self.trained_trunk
