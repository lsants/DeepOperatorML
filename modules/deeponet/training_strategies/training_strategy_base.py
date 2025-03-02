from abc import ABC, abstractmethod
import torch

class TrainingStrategy(ABC):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.phases = ['default']
        self.current_phase = 'default'
        self.prepare_before_configure = False

    @abstractmethod
    def prepare_training(self, model, **kwargs):
        pass

    def get_basis_config(self):
        pass

    def get_basis_functions(self):
        pass

    def inference_mode(self):
        pass
    
    def get_phases(self):
        return self.phases

    def update_training_phase(self, phase):
        if phase != 'default':
            raise ValueError(f"Invalid phase for current strategy: {phase}")
        self.current_phase = phase

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
        return self.loss_fn(targets, outputs)

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
        if params['LR_SCHEDULING']:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizers['default'],
                step_size=params['SCHEDULER_STEP_SIZE'],
                gamma=params['SCHEDULER_GAMMA']
            )
            return {'scheduler': scheduler}
        else:
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

    def forward(self, model, xb=None, xt=None):
        return model.output_strategy.forward(model, data_branch=xb, data_trunk=xt)

    def get_trunk_output(self, model, xt):
        return model.trunk_network(xt)

    def get_branch_output(self, model, xb):
        branch_output = model.branch_network(xb)
        return branch_output.T

    def _freeze_trunk(self, model):
        for param in model.trunk_network.parameters():
            param.requires_grad = False

    def _unfreeze_trunk(self, model):
        for param in model.trunk_network.parameters():
            param.requires_grad = True

    def _freeze_branch(self, model):
        for param in model.branch_network.parameters():
            param.requires_grad = False

    def _unfreeze_branch(self, model):
        for param in model.branch_network.parameters():
            param.requires_grad = True

    def get_basis_functions(self, **kwargs):
        xt = kwargs.get('xt')
        model = kwargs.get('model')
        n = model.n_outputs
        N_model = model.output_strategy.trunk_output_size
        N_trunk = model.n_basis_functions

        trunk_out = model.trunk_network(xt)

        if N_trunk > N_model:
            basis_functions = torch.stack(
                [trunk_out[ : , i * N_model : (i + 1) * N_model ] for i in range(n)], dim=0)
        else:
            basis_functions = trunk_out.unsqueeze(-1)
            basis_functions = torch.transpose(basis_functions, 1, 0)
        return basis_functions

    def get_coefficients(self, **kwargs):
        xb = kwargs.get('xb')
        model = kwargs.get('model')
        n = model.n_outputs
        N_model = model.output_strategy.branch_output_size
        N_branch = model.n_basis_functions

        branch_out = model.branch_network(xb)

        if N_branch > N_model:
            coefficients = torch.stack(
                [branch_out[ : , i * N_model : (i + 1) * N_model ] for i in range(n)], dim=0)
        else:
            coefficients = branch_out.unsqueeze(-1)
        return coefficients