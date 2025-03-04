# File: src/modules/deeponet/helpers/two_step_helper.py
import torch
import logging

logger = logging.getLogger(__name__)

class TwoStepHelper:
    def __init__(self, decomposition_helper, A: torch.nn.Parameter | None = None):
        """
        Initializes the TwoStepHelper.
        
        Args:
            decomposition_helper: A helper that provides a 'decompose(trunk_output)' method.
            A (optional): A matrix parameter used in the branch phase.
        """
        self.decomposition_helper = decomposition_helper
        self.A = A

    def compute_outputs(self, model, xb: torch.Tensor | None, xt: torch.Tensor | None, phase: str) -> tuple:
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
            A tuple (trunk_out, branch_out).
        """
        if phase == "trunk":
            trunk_out = model.trunk.forward(xt)
            branch_out = self.A
        elif phase == "branch":
            trunk_out = model.trunk.forward(xt)
            branch_out = model.get_branch_output(xb)
        else:
            trunk_out = model.trunk.forward(xt)
            branch_out = model.get_branch_output(xb)
        return trunk_out, branch_out

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
        if phase == "trunk":
            targets = tuple(batch[key] for key in params["OUTPUT_KEYS"])
            return loss_fn(targets, outputs)
        elif phase == "branch":
            K = model.n_basis_functions
            R = self.decomposition_helper.R
            if R is None:
                raise ValueError("TwoStepHelper: R matrix is not available; ensure trunk phase decomposition is complete.")
            # If R is computed in two blocks (2*K x 2*K), merge them.
            if R.shape[0] == 2 * K and R.shape[1] == 2 * K:
                R_first = R[:K, :K]
                R_second = R[K:, K:]
                R = torch.cat((R_first, R_second), dim=1)
            # Here, instead of calling output_handling.forward to generate targets,
            # we assume that in branch phase, synthetic targets are computed by directly combining A and R.
            # For example, we might define:
            targets = torch.matmul(R, self.A).T  # Adjust as needed based on your fusion logic.
            return loss_fn(targets, outputs)
        else:
            # Default to direct target-based computation.
            targets = tuple(batch[key] for key in params["OUTPUT_KEYS"])
            return loss_fn(targets, outputs)

    def compute_errors(self, outputs: tuple, batch: dict[str, torch.Tensor], model, params: dict, phase: str) -> dict[str, float]:
        """
        Computes error metrics in a phase-dependent manner.
        
        Follows similar logic as compute_loss.
        """
        errors = {}
        error_norm = params.get("ERROR_NORM", 2)
        if phase == "trunk":
            targets = {k: v for k, v in batch.items() if k in params["OUTPUT_KEYS"]}
            for key, target, pred in zip(params["OUTPUT_KEYS"], targets.values(), outputs):
                norm_t = torch.linalg.vector_norm(target, ord=error_norm)
                norm_e = torch.linalg.vector_norm(target - pred, ord=error_norm)
                errors[key] = (norm_e / norm_t).item() if norm_t > 0 else float("inf")
        elif phase == "branch":
            K = model.n_basis_functions
            R = self.decomposition_helper.R
            if R is None:
                raise ValueError("TwoStepHelper: R matrix is not available for error computation in branch phase.")
            if R.shape[0] == 2 * K and R.shape[1] == 2 * K:
                R_first = R[:K, :K]
                R_second = R[K:, K:]
                R = torch.cat((R_first, R_second), dim=1)
            targets = torch.matmul(R, self.A).T  # Synthetic targets.
            for key, target, pred in zip(params["OUTPUT_KEYS"], targets, outputs):
                norm_t = torch.linalg.vector_norm(target, ord=error_norm)
                norm_e = torch.linalg.vector_norm(target - pred, ord=error_norm)
                errors[key] = (norm_e / norm_t).item() if norm_t > 0 else float("inf")
        else:
            targets = {k: v for k, v in batch.items() if k in params["OUTPUT_KEYS"]}
            for key, target, pred in zip(params["OUTPUT_KEYS"], targets.values(), outputs):
                norm_t = torch.linalg.vector_norm(target, ord=error_norm)
                norm_e = torch.linalg.vector_norm(target - pred, ord=error_norm)
                errors[key] = (norm_e / norm_t).item() if norm_t > 0 else float("inf")
        return errors

    def compute_pretrained_trunk(self, model, trunk_input: torch.Tensor) -> torch.Tensor:
        """
        Computes the pretrained trunk tensor from trunk_input using the model's trunk and the decomposition helper.
        """
        trunk_output = model.trunk(trunk_input)
        pretrained = self.decomposition_helper.decompose(trunk_output)
        return pretrained

    def after_epoch(self, epoch: int, model, params: dict, **kwargs) -> None:
        """
        Optional hook to perform operations after each epoch.
        """
        # This could include logging, monitoring convergence, etc.
        logger.info(f"TwoStepHelper: Epoch {epoch} completed.")
