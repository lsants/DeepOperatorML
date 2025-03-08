import torch
from typing import TYPE_CHECKING, Optional, Callable
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

from .training_strategy_base import TrainingStrategy

class StandardTrainingStrategy(TrainingStrategy):
    def __init__(self, loss_fn: callable, output_transform: Optional[Callable] = None,**kwargs) -> None:
        super().__init__(loss_fn, output_transform)

    def get_trunk_config(self, base_trunk_config: dict[str, any]) -> dict[str, any]:
        config = base_trunk_config.copy()
        config["type"] = "trainable"
        return config

    def get_branch_config(self, base_branch_config: dict[str, any]) -> dict[str, any]:
        config = base_branch_config.copy()
        config["type"] = "trainable"
        return config

    def prepare_training(self, model: 'DeepONet', **kwargs) -> None:
        for param in model.trunk.parameters():
            param.requires_grad = True
        for param in model.branch.parameters():
            param.requires_grad = True

    def forward(self, model: 'DeepONet', xb: torch.Tensor=None, xt: torch.Tensor=None, **kwargs) -> tuple[torch.Tensor]:
        branch_out = model.branch.forward(xb)
        trunk_out = model.trunk.forward(xt)
        output = model.output_handling.forward(model, branch_out=branch_out, trunk_out=trunk_out)
        if self.output_transform is not None:
            output = tuple(self.output_transform(i) for i in output)
        return output

    def compute_loss(self, outputs, batch, model: 'DeepONet', params, **kwargs) -> float:
        targets = tuple(batch[key] for key in params["OUTPUT_KEYS"])
        return self.loss_fn(targets, outputs)

    def compute_errors(self, outputs, batch, model: 'DeepONet', params, **kwargs) -> dict[str, any]:
        errors = {}
        error_norm = params.get("ERROR_NORM", 2)
        targets = {k: v for k, v in batch.items() if k in params["OUTPUT_KEYS"]}
        for key, target, pred in zip(params["OUTPUT_KEYS"], targets.values(), outputs):
            norm_target = torch.linalg.vector_norm(target, ord=error_norm)
            norm_error = torch.linalg.vector_norm(target - pred, ord=error_norm)
            errors[key] = (norm_error / norm_target).item() if norm_target > 0 else float("inf")
        return errors
