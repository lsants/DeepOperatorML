from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

import torch

class TrainingStrategy(ABC):
    def __init__(self, loss_fn: callable, **kwargs) -> None:
        self.loss_fn: callable = loss_fn
        self.phases: list[str] = ['default']
        self.current_phase: str = 'default'
        self.prepare_before_configure: bool = False
        self.inference = False

    @abstractmethod
    def prepare_training(self, model: 'DeepONet', **kwargs) -> None:
        pass

    def inference_mode(self) -> None:
        self.inference = True
    
    def get_phases(self) -> list[str]:
        return self.phases

    def update_training_phase(self, phase: str) -> None:
        if phase != 'default':
            raise ValueError(f"Invalid phase for current strategy: {phase}")
        self.current_phase = phase

    def prepare_for_phase(self, model, **kwargs) -> None:
        pass

    def get_epochs(self, params: dict[str, any]) -> list[int]:
        return [params['EPOCHS']]
    
    def before_epoch(self, epoch: int, model: 'DeepONet', params: dict[str, any]) -> None:
        pass

    def after_epoch(self, epoch: int, model: 'DeepONet', params: dict[str, any], **kwargs) -> None:
        return {}

    def compute_loss(self, outputs: tuple[torch.Tensor], batch: dict[str, torch.Tensor], model: 'DeepONet', params: dict[str, any], **kwargs) -> float:
        targets = tuple(batch[key] for key in params['OUTPUT_KEYS'])
        return self._compute_loss_default(outputs, targets)

    def _compute_loss_default(self, outputs, targets) -> float:
        return self.loss_fn(targets, outputs)

    def compute_errors(self, outputs: tuple[torch.Tensor], batch: dict[str, torch.Tensor], model: 'DeepONet', params: dict[str, any], **kwargs) -> dict[str, any]:
        errors = {}
        targets = {k:v for k,v in batch.items() if k in params['OUTPUT_KEYS']}
        for key, target, pred in zip(params['OUTPUT_KEYS'], targets.values(), outputs):
            if key in params['OUTPUT_KEYS']:
                error = (
                    torch.linalg.vector_norm(target - pred, ord=params['ERROR_NORM'])
                    / torch.linalg.vector_norm(target, ord=params['ERROR_NORM'])
                ).item()
                errors[key] = error
        self.errors = errors
        return errors

    def get_optimizers(self, model: 'DeepONet', params: dict[str, any]) -> dict[str, torch.optim.Optimizer]:
        optimizer = torch.optim.Adam(list(model.parameters()), 
                                     lr=params['LEARNING_RATE'], 
                                     weight_decay=params['L2_REGULARIZATION'])
        return {self.current_phase: optimizer}

    def get_schedulers(self, optimizers: torch.optim.Optimizer, params: dict[str, any]) -> dict:
        if params['LR_SCHEDULING']:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizers['default'],
                step_size=params['SCHEDULER_STEP_SIZE'],
                gamma=params['SCHEDULER_GAMMA']
            )
            return {'scheduler': scheduler}
        else:
            return {}

    def can_validate(self) -> bool:
        return True
    
    def zero_grad(self, optimizers: dict[str, torch.optim.Optimizer]) -> None:
        for _, optimizer in optimizers.items():
            if optimizer is not None:
                optimizer.zero_grad()

    def step(self,optimizers: dict[str, torch.optim.Optimizer]) -> None:
        for _, optimizer in optimizers.items():
            if optimizer is not None:
                optimizer.step()

    def step_schedulers(self, schedulers: dict[str, torch.optim.Optimizer]) -> None:
        for _, scheduler in schedulers.items():
            if scheduler is not None:
                scheduler.step()

    def forward(self, model, xb=None, xt=None) -> tuple[torch.Tensor]:
        return model.output_strategy.forward(model, data_branch=xb, data_trunk=xt)

    def get_trunk_output(self, model: 'DeepONet', xt: torch.Tensor) -> torch.Tensor:
        return model.trunk_network(xt)

    def get_branch_output(self, model: 'DeepONet', xb: torch.Tensor) -> torch.Tensor:
        branch_output = model.branch_network(xb)
        return branch_output.T

    def _freeze_trunk(self, model: 'DeepONet') -> None:
        for param in model.trunk_network.parameters():
            param.requires_grad = False

    def _unfreeze_trunk(self, model: 'DeepONet') -> None:
        for param in model.trunk_network.parameters():
            param.requires_grad = True

    def _freeze_branch(self, model: 'DeepONet') -> None:
        for param in model.branch_network.parameters():
            param.requires_grad = False

    def _unfreeze_branch(self, model: 'DeepONet') -> None:
        for param in model.branch_network.parameters():
            param.requires_grad = True

    def get_basis_callables(self, **kwargs) -> torch.Tensor:
        xt = kwargs.get('xt')
        model = kwargs.get('model')
        n = model.n_outputs
        N_model = model.output_strategy.trunk_output_size
        N_trunk = model.n_basis_callables

        trunk_out = model.trunk_network(xt)

        if N_trunk > N_model:
            basis_callables = torch.stack(
                [trunk_out[ : , i * N_model : (i + 1) * N_model ] for i in range(n)], dim=0)
        else:
            basis_callables = trunk_out.unsqueeze(-1)
            basis_callables = torch.transpose(basis_callables, 1, 0)
        return basis_callables

    def get_coefficients(self, **kwargs) -> torch.Tensor:
        xb = kwargs.get('xb')
        model = kwargs.get('model')
        n = model.n_outputs
        N_model = model.output_strategy.branch_output_size
        N_branch = model.n_basis_callables

        branch_out = model.branch_network(xb)

        if N_branch > N_model:
            coefficients = torch.stack(
                [branch_out[ : , i * N_model : (i + 1) * N_model ] for i in range(n)], dim=0)
        else:
            coefficients = branch_out.unsqueeze(-1)
        return coefficients