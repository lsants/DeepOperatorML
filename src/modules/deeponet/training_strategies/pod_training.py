# File: src/modules/deeponet/training_strategies/pod_training_strategy.py
import torch
from .training_strategy_base import TrainingStrategy
from ..deeponet import DeepONet
import logging

logger = logging.getLogger(__name__)

class PODTrainingStrategy(TrainingStrategy):
    """
    PODTrainingStrategy assumes that the output strategy is responsible for computing 
    the basis functions from data and reconfiguring the trunk as a fixed-tensor component.
    
    This strategy then focuses on:
      - Performing a forward pass using the fixed trunk.
      - Computing the loss and error based on the final prediction.
      - Operating in a single-phase (no phase switching needed).
    """
    def __init__(self, loss_fn: callable, inference: bool, **kwargs) -> None:
        super().__init__(loss_fn)
        self.inference = inference
    
    def prepare_training(self, model: DeepONet, **kwargs) -> None:
        """
        For POD, preparation may include verifying that the trunk has been reconfigured
        as a fixed tensor (i.e. the basis functions have been computed by the output strategy).
        """
        # Optionally verify that model.trunk has the expected attribute or type.
        if not hasattr(model.trunk, "get_basis"):
            raise ValueError("The trunk component is not configured correctly for POD training.")
        logger.info("PODTrainingStrategy: Model trunk is configured for POD.")
    
    def forward(self, model: DeepONet, xb: torch.Tensor | None = None, xt: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        In POD training, the trunk is expected to be fixed (returning precomputed basis functions)
        and the branch network produces coefficients. The output strategy fuses these to produce the prediction.
        """
        # Here, xt may be None because the trunk's forward (or get_basis) returns the fixed tensor.
        branch_out = model.branch.forward(xb)
        trunk_out = model.trunk.forward()
        dot_product = model.output_handling.forward(model, data_branch=branch_out, data_trunk=trunk_out)
        if model.output_handling.BASIS_CONFIG == 'single':
            output = tuple(x + model.trunk.mean for x in dot_product)
        else:
            output = tuple(dot_product[i] + model.trunk.mean[i : i + 1] for i in range(model.n_outputs))
        return output
    
    def compute_loss(self, outputs: tuple[torch.Tensor], batch: dict[str, torch.Tensor], model: DeepONet, params: dict, **kwargs) -> float:
        """
        Computes the loss by comparing model outputs with targets extracted from the batch.
        """
        # Assume targets are stored under keys defined in params["OUTPUT_KEYS"]
        targets = tuple(batch[key] for key in params["OUTPUT_KEYS"])
        return self.loss_fn(targets, outputs)
    
    def compute_errors(self, outputs: tuple[torch.Tensor], batch: dict[str, torch.Tensor], model: DeepONet, params: dict, **kwargs) -> dict[str, float]:
        """
        Computes errors using a relative norm between predictions and targets.
        """
        errors = {}
        error_norm = params.get("ERROR_NORM", 2)
        targets = {k: v for k, v in batch.items() if k in params["OUTPUT_KEYS"]}
        for key, target, pred in zip(params["OUTPUT_KEYS"], targets.values(), outputs):
            norm_target = torch.linalg.vector_norm(target, ord=error_norm)
            norm_error = torch.linalg.vector_norm(target - pred, ord=error_norm)
            errors[key] = (norm_error / norm_target).item() if norm_target > 0 else float("inf")
        return errors
    
    def get_trunk_config(self, base_trunk_config: dict) -> dict:
        config = base_trunk_config.copy()
        if self.inference:
            config["type"] = "data"
            if self.basis is None or self.mean is None:
                raise ValueError("PODTrainingStrategy: Pretrained trunk tensor not available in inference mode.")
            config["data"] = self.pretrained_trunk_tensor
        else:
            config["type"] = "trainable"
        return config

    def get_branch_config(self, base_branch_config: dict) -> dict:
        config = base_branch_config.copy()
        config["type"] = "trainable"
        return config
    
    def update_training_phase(self, phase: str) -> None:
        # For single-phase, simply log that POD uses a default phase.
        logger.info("StandardTrainingStrategy: Using single-phase training.")
    
    def prepare_for_phase(self, model: DeepONet, **kwargs) -> None:
        pass
    
    def after_epoch(self, epoch: int, model: DeepONet, params: dict, **kwargs) -> None:
        pass
