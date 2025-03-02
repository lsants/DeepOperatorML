import torch
import logging
from .training_strategy_base import TrainingStrategy

logger = logging.getLogger(__name__)


class TwoStepTrainingStrategy(TrainingStrategy):
    def __init__(self, loss_fn, device, precision, train_dataset_length=None):
        super().__init__(loss_fn)
        self.train_dataset_length = train_dataset_length
        self.device = device
        self.precision = precision
        if not self.train_dataset_length:
            logger.warning(
                "Initializing the model without A matrix. Only do this if you're doing inference.")
        self.A = None
        self.Q = None
        self.R = None
        self.T = None
        self.trained_trunk = None
        self.phases = ['trunk', 'branch', 'final']
        self.current_phase = self.phases[0]
        self.prepare_before_configure = False

    def get_epochs(self, params):
        return [params['TRUNK_TRAIN_EPOCHS'], params['BRANCH_TRAIN_EPOCHS']]

    def update_training_phase(self, phase, **kwargs):
        self.current_phase = phase
        logger.info(f'Current phase: {self.current_phase}')

    def prepare_training(self, model, **kwargs):
        branch_output_size = getattr(
            model.output_strategy, 'branch_output_size')
        if self.train_dataset_length and self.A is None:
            A_dim = (branch_output_size, self.train_dataset_length)

            logger.info(
                f"A matrix's dimensions: {branch_output_size, self.train_dataset_length}")
            logger.info(
                f"Creating trainable matrix")

            self.A = torch.nn.Parameter(torch.randn(A_dim)).to(device=self.device, 
                                                               dtype=self.precision)
            torch.nn.init.kaiming_uniform_(self.A)

    def prepare_for_phase(self, model, **kwargs):
        params = kwargs.get('model_params')
        xt = kwargs.get('train_batch')
        self._set_phase_params(model, self.current_phase)
        if self.current_phase == 'branch' and self.trained_trunk is None:
            self.update_q_r_t_matrices(model, params, xt)
            with torch.no_grad():
                self.branch_matrix = {
                    'trunk_matrix': self.R,
                    'branch_matrix': self.A
                }

    def _set_phase_params(self, model, phase):
        if phase == 'trunk':
            self._freeze_branch(model)
            self._unfreeze_trunk(model)
        elif phase == 'branch':
            self._freeze_trunk(model)
            self._unfreeze_branch(model)

    def _freeze_trunk(self, model):
        for param in model.trunk_network.parameters():
            param.requires_grad = False
        self.A.requires_grad = False

    def _unfreeze_trunk(self, model):
        for param in model.trunk_network.parameters():
            param.requires_grad = True
        self.A.requires_grad = True

    def _freeze_branch(self, model):
        for param in model.branch_network.parameters():
            param.requires_grad = False

    def _unfreeze_branch(self, model):
        for param in model.branch_network.parameters():
            param.requires_grad = True

    def compute_loss(self, outputs, batch, model, params, **kwargs):
        if self.current_phase == 'trunk':
            targets = tuple(batch[key] for key in params['OUTPUT_KEYS'])
            loss = self.loss_fn(targets, outputs)
        elif self.current_phase == 'branch':
            targets = model.output_strategy.forward(
                model,
                data_branch=None,
                data_trunk=None,
                matrix_branch=self.A,
                matrix_trunk=self.R
            )

            loss = self.loss_fn(targets, outputs)
        elif self.current_phase == 'final':
            targets = tuple(batch[key] for key in params['OUTPUT_KEYS'])
            loss = self.loss_fn(targets, outputs)
        else:
            raise ValueError(f"Unknown training phase: {self.current_phase}")
        return loss

    def compute_errors(self, outputs, batch, model, params, **kwargs):
        errors = {}
        if self.current_phase in ['trunk', 'final']:
            targets = {k: v for k, v in batch.items(
            ) if k in params['OUTPUT_KEYS']}
            for key, target, pred in zip(params['OUTPUT_KEYS'], targets.values(), outputs):
                if key in params['OUTPUT_KEYS']:
                    error = (
                        torch.linalg.vector_norm(
                            target - pred, ord=params['ERROR_NORM'])
                        / torch.linalg.vector_norm(target, ord=params['ERROR_NORM'])
                    ).item()
                    errors[key] = error
        elif self.current_phase == 'branch':
            targets = model.output_strategy.forward(
                model,
                data_branch=None,
                data_trunk=None,
                matrix_branch=self.A,
                matrix_trunk=self.R
            )
            for _, (key, target, pred) in enumerate(zip(params['OUTPUT_KEYS'], targets, outputs)):
                error = (
                    torch.linalg.vector_norm(
                        target - pred, ord=params['ERROR_NORM'])
                    / torch.linalg.vector_norm(target, ord=params['ERROR_NORM'])
                ).item()
                errors[key] = error
        else:
            raise ValueError(f"Unknown training phase: {self.current_phase}")
        return errors

    def can_validate(self):
        return False

    def after_epoch(self, epoch, model, params, **kwargs):
        if self.current_phase == 'trunk' and epoch + 1 == params['TRUNK_TRAIN_EPOCHS']:
            logger.debug(
                f"THIS RAN BECAUSE PHASE ({self.current_phase}) SHOULD BE TRUNK AND NEXT EPOCH ({epoch + 1}) IS {params['TRUNK_TRAIN_EPOCHS']}")
            train_batch = kwargs.get('train_batch')
            self.update_q_r_t_matrices(model, params, train_batch)
            logger.info(
                f"Trunk matrix updated and phase transition triggered at epoch {epoch + 1}")

    def get_optimizers(self, model, params):
        optimizers = {}

        trunk_params = [i for i in model.trunk_network.parameters()]
        trunk_params.append(self.A)

        optimizers['trunk'] = torch.optim.Adam(
            trunk_params, lr=params['TRUNK_LEARNING_RATE'], weight_decay=params['L2_REGULARIZATION'])

        branch_params = [i for i in model.branch_network.parameters()]

        optimizers['branch'] = torch.optim.Adam(
            branch_params, lr=params['BRANCH_LEARNING_RATE'], weight_decay=params['L2_REGULARIZATION'])
        return optimizers

    def get_schedulers(self, optimizers, params):
        schedulers = {}
        if params["LR_SCHEDULING"]:
            schedulers['trunk'] = torch.optim.lr_scheduler.StepLR(
                optimizers['trunk'],
                step_size=params['TRUNK_SCHEDULER_STEP_SIZE'],
                gamma=params['TRUNK_SCHEDULER_GAMMA']
            )
            schedulers['branch'] = torch.optim.lr_scheduler.StepLR(
                optimizers['branch'],
                step_size=params['BRANCH_SCHEDULER_STEP_SIZE'],
                gamma=params['BRANCH_SCHEDULER_GAMMA']
            )
        return schedulers

    def zero_grad(self, optimizers):
        if self.current_phase == 'trunk':
            optimizers['trunk'].zero_grad()
        elif self.current_phase == 'branch':
            optimizers['branch'].zero_grad()

    def step(self, optimizers):
        if self.current_phase == 'trunk':
            optimizers['trunk'].step()
        elif self.current_phase == 'branch':
            optimizers['branch'].step()

    def step_schedulers(self, schedulers):
        if self.current_phase == 'trunk':
            schedulers['trunk'].step()
        elif self.current_phase == 'branch':
            schedulers['branch'].step()

    def get_trunk_output(self, model, xt):
        if xt is not None:
            trunk_output = model.trunk_network(xt)
        return trunk_output

    def get_branch_output(self, model, xb):
        branch_output = model.branch_network(xb)
        return branch_output.T

    def forward(self, model, xb=None, xt=None):
        if self.current_phase == 'trunk':
            input_branch = self.A
            input_trunk = xt
            return model.output_strategy.forward(model,
                                                 data_branch=None,
                                                 data_trunk=input_trunk,
                                                 matrix_branch=input_branch,
                                                 matrix_trunk=None)
        elif self.current_phase == 'branch':
            input_branch = xb
            return model.output_strategy.forward(model,
                                                 data_branch=input_branch,
                                                 data_trunk=None,
                                                 matrix_branch=None,
                                                 matrix_trunk=None)
        else:
            input_branch = xb
            input_trunk = self.trained_trunk
            return model.output_strategy.forward(model,
                                                 data_branch=input_branch,
                                                 data_trunk=None,
                                                 matrix_branch=None,
                                                 matrix_trunk=input_trunk)

    def get_basis_functions(self, **kwargs):
        trunk_outputs = self.trained_trunk
        model = kwargs.get('model')
        N_model = model.output_strategy.trunk_output_size
        N_trunk = model.n_basis_functions
        n = model.n_outputs

        if N_trunk > N_model:
            basis_functions = torch.stack(
                [trunk_outputs[ : , i * N_model : (i + 1) * N_model ] for i in range(n)], dim=0)
        else:
            basis_functions = trunk_outputs.unsqueeze(-1)
            basis_functions = torch.transpose(basis_functions, 1, 0)
        return basis_functions

    def update_q_r_t_matrices(self, model, params,  xt):
        with torch.no_grad():
            decomposition = params.get('TRUNK_DECOMPOSITION')
            phi = model.trunk_network(xt)

            if decomposition.lower() == 'qr':
                logger.info(f"Decomposition using QR factorization...")
                Q, R = torch.linalg.qr(phi)
                self.Q = Q
                self.R = R

            if decomposition.lower() == 'svd':
                logger.info(f"Decomposition using SVD...")
                Q, Sd, Vd = torch.linalg.svd(phi, full_matrices=False)
                R = torch.diag(Sd) @ Vd
                self.Q = Q
                self.R = R

            self.T = torch.linalg.inv(R)

            self.trained_trunk = self.Q @ self.R @ self.T

            logger.info(
                f"Q shape: {self.Q.shape}, R shape: {self.R.shape}, T shape: {self.T.shape}")
            logger.info(f"Reconstructed Phi shape: {(self.Q @ self.R @ self.T).shape}")
            logger.info(
                f"Q @ R == Phi check: {torch.allclose(self.Q @ self.R, phi, atol=1e-5)}")
            
            if self.Q is None or self.R is None or self.T is None:
                raise ValueError(
                    f"Trunk decomposition failed. At least one of the matrix wasn't stored.")
            else:
                logger.info(
                    f"Trunk decomposed successfully. \nMoving on to second step...")

    def set_matrices(self, **kwargs):
        self.Q = kwargs.get('Q')
        self.R = kwargs.get('R')
        self.T = kwargs.get('T')

        if self.Q is None:
            raise ValueError("ERROR: Q matrix couldn't be assigned.")
        if self.R is None:
            raise ValueError("ERROR: R matrix couldn't be assigned.")
        if self.T is None:
            raise ValueError("ERROR: T matrix couldn't be assigned.")
        
        self.trained_trunk = self.Q @ self.R @ self.T
        
        logger.info(
            f"Set trained trunk (shaped {(self.trained_trunk.shape[0], self.trained_trunk.shape[1])}) for inference.")

    def inference_mode(self):
        self.current_phase = 'final'
