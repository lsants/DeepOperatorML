from __future__ import annotations
import torch
import logging
from .....exceptions import MissingSettingError
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class TwoStepHelper:
    def __init__(self, decomposition_helper, device: str, precision: torch.dtype) -> None:
        """
        Initializes the TwoStepHelper.
        
        Args:
            decomposition_helper: A helper that provides a 'decompose(trunk_output)' method.
            A (optional): A matrix parameter used in the branch phase.
        """
        self.decomposition_helper = decomposition_helper
        self.device = device
        self.precision = precision

    def set_A_matrix(self, branch_batch_size: int, branch_output_size: int) -> None:
        A_dims = (branch_batch_size, branch_output_size)
        A = torch.randn(A_dims).to(device=self.device,
                                   dtype=self.precision)
        return A
    
    def compute_synthetic_targets(self, model: 'DeepONet') -> tuple[torch.Tensor]:
        K = model.n_basis_functions
        n = model.n_outputs
        R = self.decomposition_helper.R
        A = model.training_strategy.A
        if R is None or A is None:
            raise ValueError("R or A matrix are not available; ensure trunk phase decomposition is complete.")
        
        # If there are multiple trunks (size of R = n * number of basis functions), merge the A matrix
        if R.shape == (n * K, n * K):
            blocks = [R[i * K : (i + 1) * K , i * K : (i + 1) * K] for i in range(n)]
            R = torch.cat(blocks, dim=1)
            targets = tuple(model.output_handling.forward(model, branch_out=model.training_strategy.A, trunk_out=R))
        else:
            targets = model.output_handling.forward(model, branch_out=model.training_strategy.A, trunk_out=R)
        return targets

    def compute_outputs(self, model: 'DeepONet', xb: torch.Tensor | None, xt: torch.Tensor | None, phase: str) -> tuple[torch.Tensor]:
        """
        Computes trunk and branch outputs based on the current phase.
        
        For the trunk phase:
          - The trunk output is computed normally.
          - The branch output is provided via A (if available) or computed normally.
          
        For the branch phase:
          - Both trunk and branch outputs are computed from their respective components.
          
        Args:
            model: The DeepONet model.
            xb: Input for branch component.
            xt: Input for trunk component.
            phase: Current phase ("trunk" or "branch").
        
        Returns:
            A tuple (branch_out, trunk_out).
        """
        if not model.training_strategy.inference:
            if phase == "trunk":
                trunk_out = model.trunk.forward(xt)
                branch_out = model.trunk.A
                return branch_out, trunk_out 
            elif phase == "branch":
                branch_out = model.branch.forward(xb)
                return branch_out, ...
            else:
                raise ValueError("Invalid training phase.")
        else:
            trunk_out = model.trunk.forward(xt)
            branch_out = model.branch.forward(xb)
            return branch_out, trunk_out 

    def compute_loss(self, outputs: tuple, batch: dict[str, torch.Tensor], model, params: dict, phase: str, loss_fn: callable) -> float:
        """
        Computes loss in a phase-dependent manner.
        
        - In "trunk" phase, loss is computed directly from targets extracted from the batch.
        - In "branch" phase, synthetic targets are computed using the pretrained trunk tensor.
        
        Args:
            outputs: Model outputs.
            batch: Dictionary containing batch data.
            model: The DeepONet model.
            params: Training parameters, including OUTPUT_KEYS.
            phase: Current phase ("trunk" or "branch").
            loss_fn: A loss function that accepts (targets, outputs).
        
        Returns:
            The computed loss (a float).
        """
        if not model.training_strategy.inference:
            if phase == "trunk":
                targets = tuple(batch[key] for key in params["OUTPUT_KEYS"])
            elif phase == "branch":
                targets = self.compute_synthetic_targets(model)
            else:
                raise ValueError("Invalid training phase.")
        else:
            targets = tuple(batch[key] for key in params["OUTPUT_KEYS"])
        return loss_fn(targets, outputs)

    def compute_errors(self, outputs: tuple, batch: dict[str, torch.Tensor], model, params: dict, phase: str) -> dict[str, float]:
        """
        Computes error metrics in a phase-dependent manner.
        
        Follows similar logic as compute_loss.
        """
        errors = {}
        error_norm = params.get("ERROR_NORM")
        if not error_norm:
            raise MissingSettingError("Norm for error computation is undefined. Check .yaml file.")

        if not model.training_strategy.inference:
            if phase == "trunk":
                targets = {k: v for k, v in batch.items() if k in params["OUTPUT_KEYS"]}
            elif phase == "branch":
                targets = self.compute_synthetic_targets(model)
                for key, target, pred in zip(params["OUTPUT_KEYS"], targets, outputs):
                    norm_t = torch.linalg.vector_norm(target, ord=error_norm)
                    norm_e = torch.linalg.vector_norm(target - pred, ord=error_norm)
                    errors[key] = (norm_e / norm_t).item() if norm_t > 0 else float("inf")
                return errors
            else:
                raise ValueError("Invalid training phase.")
        else:
            targets = {k: v for k, v in batch.items() if k in params["OUTPUT_KEYS"]}
        for key, target, pred in zip(params["OUTPUT_KEYS"], targets.values(), outputs):
            norm_t = torch.linalg.vector_norm(target, ord=error_norm)
            norm_e = torch.linalg.vector_norm(target - pred, ord=error_norm)
            errors[key] = (norm_e / norm_t).item() if norm_t > 0 else float("inf")
        return errors

    def compute_trained_trunk(self, model, params: dict, trunk_input: torch.Tensor) -> torch.Tensor:
        """
        Computes the pretrained trunk tensor from trunk_input using the model's trunk and the decomposition helper.
        """
        pretrained = self.decomposition_helper.decompose(model, params, trunk_input)
        return pretrained
    
    def after_epoch(self, epoch: int, model, params: dict, **kwargs) -> None:
        """
        Optional hook to perform operations after each epoch.
        """
        # This could include logging, monitoring convergence, etc.
        # logger.info(f"TwoStepHelper: Epoch {epoch} completed.")