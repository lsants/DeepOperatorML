import torch
from typing import TYPE_CHECKING, Callable, Iterable, Any
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet
from .training_strategy_base import TrainingStrategy

class StandardTrainingStrategy(TrainingStrategy):
    def __init__(self, loss_fn: Callable[[Iterable[torch.Tensor], Iterable[torch.Tensor]]], **kwargs) -> None:
        super().__init__(loss_fn)

    def get_trunk_config(self, trunk_config: dict[str, Any]) -> dict[str, Any]:
        config = trunk_config.copy()
        config["type"] = "trainable"
        return config

    def get_branch_config(self, branch_config: dict[str, Any]) -> dict[str, Any]:
        config = branch_config.copy()
        config["type"] = "trainable"
        return config

    def prepare_for_training(self, model: 'DeepONet', **kwargs) -> None:
        for param in model.trunk.parameters():
            param.requires_grad = True
        for param in model.branch.parameters():
            param.requires_grad = True

    def forward(self, 
                model: 'DeepONet', 
                branch_input: torch.Tensor | None = None, 
                trunk_input: torch.Tensor | None = None, 
                **kwargs) -> tuple[torch.Tensor]:
        branch_out = model.branch.forward(branch_input)
        trunk_out = model.trunk.forward(trunk_input)
        output = model.output_handling.forward(
            model=model, branch_out=branch_out, trunk_out=trunk_out)
        return output

    def compute_loss(self, outputs, batch, model: 'DeepONet', **kwargs:dict[str, Any]) -> torch.Tensor:
        training_params = kwargs.get('training_params')
        targets = tuple(batch[key] for key in training_params["OUTPUT_KEYS"])
        return self.loss_fn(targets, outputs)

    def compute_errors(self, outputs, batch, model: 'DeepONet', **kwargs: Any) -> dict[str, Any]:
        errors = {}
        training_params = kwargs.get('training_params')
        error_norm = training_params.get("ERROR_NORM", 2)
        if training_params:
            targets = {k: v for k, v in batch.items(
            ) if k in training_params["OUTPUT_KEYS"]}
        for key, target, pred in zip(training_params["OUTPUT_KEYS"], targets.values(), outputs):
            norm_target = torch.linalg.vector_norm(target, ord=error_norm)
            norm_error = torch.linalg.vector_norm(
                target - pred, ord=error_norm)
            errors[key] = (
                norm_error / norm_target).item() if norm_target > 0 else float("inf")
        return errors
