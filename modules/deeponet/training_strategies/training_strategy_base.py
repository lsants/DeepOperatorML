from abc import ABC, abstractmethod
import torch
from ..optimization.loss_complex import loss_complex

class TrainingStrategy(ABC):
    def __init__(self):
        self.phases = ['default']
        self.current_phase = 'default'

    @abstractmethod
    def prepare_training(self, model):
        pass

    def inference_mode(self):
        pass
    
    def get_phases(self):
        return self.phases

    def update_training_phase(self, phase):
        if phase not in self.phases:
            raise ValueError(f"Unknown phase: {phase}")
        self.current_phase = phase
        print(f"Updated to phase: {self.current_phase}")

    def prepare_for_phase(self, model, **kwargs):
        pass

    def get_epochs(self, params):
        return [params['EPOCHS']]
    
    def before_epoch(self, epoch, model, params):
        pass

    def after_epoch(self, epoch, model, params, **kwargs):
        return {}

    def compute_loss(self, outputs, batch, model, params, **kwargs):
        targets = tuple(batch[key] for key in params['OUTPUT_KEYS'])
        return self._compute_loss_default(outputs, targets)

    def _compute_loss_default(self, outputs, targets):
        return loss_complex(targets, outputs)

    def compute_errors(self, outputs, batch, model, params, **kwargs):
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

    def get_optimizers(self, model, params):
        optimizer = torch.optim.Adam(list(model.parameters()), 
                                     lr=params['LEARNING_RATE'], 
                                     weight_decay=params['L2_REGULARIZATION'])
        return {self.current_phase: optimizer}

    def get_schedulers(self, optimizers, params):
        if 'optimizer' in optimizers and optimizers['optimizer']:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizers['optimizer'],
                step_size=params['SCHEDULER_STEP_SIZE'],
                gamma=params['SCHEDULER_GAMMA']
            )
            return {'scheduler': scheduler}
        return {}

    def can_validate(self):
        return True
    
    def zero_grad(self, optimizers):
        for _, optimizer in optimizers.items():
            if optimizer is not None:
                optimizer.zero_grad()

    def step(self, optimizers):
        for _, optimizer in optimizers.items():
            if optimizer is not None:
                optimizer.step()

    def step_schedulers(self, schedulers):
        for _, scheduler in schedulers.items():
            if scheduler is not None:
                scheduler.step()

    def forward(self, model, xb, xt):
        return model.output_strategy.forward(model, xb, xt)

    def get_trunk_output(self, model, i, xt_i):
        return model.trunk_networks[i % len(model.trunk_networks)](xt_i)

    def get_branch_output(self, model, i, xb_i):
        branch_output = model.branch_networks[i % len(model.branch_networks)](xb_i)
        return branch_output.T

    def _freeze_trunk(self, model):
        for trunk in model.trunk_networks:
            for param in trunk.parameters():
                param.requires_grad = False

    def _unfreeze_trunk(self, model):
        for trunk in model.trunk_networks:
            for param in trunk.parameters():
                param.requires_grad = True

    def _freeze_branch(self, model):
        for branch in model.branch_networks:
            for param in branch.parameters():
                param.requires_grad = False

    def _unfreeze_branch(self, model):
        for branch in model.branch_networks:
            for param in branch.parameters():
                param.requires_grad = True