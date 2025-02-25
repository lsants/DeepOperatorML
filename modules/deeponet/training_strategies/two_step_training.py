import torch
import logging
from .training_strategy_base import TrainingStrategy
from ..optimization.loss_complex import loss_complex

logger = logging.getLogger(__name__)
class TwoStepTrainingStrategy(TrainingStrategy):
    def __init__(self, train_dataset_length=None):
        self.train_dataset_length = train_dataset_length
        if not self.train_dataset_length:
            logger.warning("Initializing the model without A matrix. Only do this if you're doing inference.")
        self.A_list = None
        self.Q_list = []
        self.R_list = []
        self.T_list = []
        self.trained_trunk_list = []
        self.phases = ['trunk', 'branch', 'final']
        self.current_phase = self.phases[0]
        self.prepare_before_configure = False

    def get_epochs(self, params):
            return [params['TRUNK_TRAIN_EPOCHS'], params['BRANCH_TRAIN_EPOCHS']]

    def update_training_phase(self, phase, **kwargs):
        self.current_phase = phase
        logger.info(f'Current phase: {self.current_phase}')

    def prepare_training(self, model, **kwargs):
        branch_output_size = getattr(model.output_strategy, 'branch_output_dim')
        if self.train_dataset_length and not self.A_list:
            A_dim = (branch_output_size, self.train_dataset_length)

            logger.debug(f"A matrix's dimensions: {branch_output_size, self.train_dataset_length}")
            logger.debug(f"Creating {model.output_strategy.num_branches} trainable matrices")

            self.A_list = torch.nn.ParameterList([
                torch.nn.Parameter(torch.randn(A_dim))
                for _ in range(model.output_strategy.num_branches)
            ])
            for A in self.A_list:
                torch.nn.init.kaiming_uniform_(A)
    
    def prepare_for_phase(self, model, **kwargs):
        params = kwargs.get('model_params')
        xt = kwargs.get('train_batch')
        self._set_phase_params(model, self.current_phase)
        if self.current_phase == 'branch' and not self.trained_trunk_list:
            self.update_q_r_t_matrices(model, params, xt)
            with torch.no_grad():
                self.branch_matrices = {
                    'trunk_matrices': self.R_list,
                    'branch_matrices': self.A_list
                }

    def _set_phase_params(self, model, phase):
        if phase == 'trunk':
            self._freeze_branch(model)
            self._unfreeze_trunk(model)
        elif phase == 'branch':
            self._freeze_trunk(model)
            self._unfreeze_branch(model)

    def _freeze_trunk(self, model):
        for trunk in model.trunk_networks:
            for param in trunk.parameters():
                param.requires_grad = False
        for A in self.A_list:
            A.requires_grad = False

    def _unfreeze_trunk(self, model):
        for trunk in model.trunk_networks:
            for param in trunk.parameters():
                param.requires_grad = True
        for A in self.A_list:
            A.requires_grad = True

    def _freeze_branch(self, model):
        for branch in model.branch_networks:
            for param in branch.parameters():
                param.requires_grad = False

    def _unfreeze_branch(self, model):
        for branch in model.branch_networks:
            for param in branch.parameters():
                param.requires_grad = True
   
    def compute_loss(self, outputs, batch, model, params, **kwargs):
        if self.current_phase == 'trunk':
            targets = tuple(batch[key] for key in params['OUTPUT_KEYS'])
            loss = loss_complex(targets, outputs)
        elif self.current_phase == 'branch':
            targets = model.output_strategy.forward(
                model, 
                data_branch=None, 
                data_trunk=None, 
                matrices_branch=self.A_list, 
                matrices_trunk=self.R_list
            )

            loss = loss_complex(targets, outputs)
        elif self.current_phase == 'final':
            targets = tuple(batch[key] for key in params['OUTPUT_KEYS'])
            loss = loss_complex(targets, outputs)
        else:
            raise ValueError(f"Unknown training phase: {self.current_phase}")
        return loss

    def compute_errors(self, outputs, batch, model, params, **kwargs):
        errors = {}
        if self.current_phase in ['trunk', 'final']:
            targets = {k:v for k,v in batch.items() if k in params['OUTPUT_KEYS']}
            for key, target, pred in zip(params['OUTPUT_KEYS'], targets.values(), outputs):
                if key in params['OUTPUT_KEYS']:
                    error = (
                        torch.linalg.vector_norm(target - pred, ord=params['ERROR_NORM'])
                        / torch.linalg.vector_norm(target, ord=params['ERROR_NORM'])
                    ).item()
                    errors[key] = error
        elif self.current_phase == 'branch':
            targets = model.output_strategy.forward(
                model, 
                data_branch=None, 
                data_trunk=None, 
                matrices_branch=self.A_list, 
                matrices_trunk=self.R_list
            )
            for _, (key, target, pred) in enumerate(zip(params['OUTPUT_KEYS'], targets, outputs)):
                error = (
                    torch.linalg.vector_norm(target - pred, ord=params['ERROR_NORM'])
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
            logger.debug(f"THIS RAN BECAUSE PHASE ({self.current_phase}) SHOULD BE TRUNK AND NEXT EPOCH ({epoch + 1}) IS {params['TRUNK_TRAIN_EPOCHS']}")
            train_batch = kwargs.get('train_batch')
            self.update_q_r_t_matrices(model, params, train_batch)
            logger.info(f"Trunk matrices updated and phase transition triggered at epoch {epoch + 1}")

    def get_optimizers(self, model, params):
        optimizers = {}

        trunk_params = []
        for trunk in model.trunk_networks:
            trunk_params += list(trunk.parameters())
        trunk_params += list(self.A_list)
        optimizers['trunk'] = torch.optim.Adam(
            trunk_params, lr=params['TRUNK_LEARNING_RATE'], weight_decay=params['L2_REGULARIZATION'])
        
        branch_params = []
        for branch in model.branch_networks:
            branch_params += list(branch.parameters())
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

    def get_trunk_output(self, model, i, xt_i):
        return model.trunk_networks[i % len(model.trunk_networks)](xt_i)

    def get_branch_output(self, model, i, xb_i):
        branch_output = model.branch_networks[i % len(model.branch_networks)](xb_i)
        return branch_output.T

    def forward(self, model, xb=None, xt=None):
        if self.current_phase == 'trunk':
            input_branch = self.A_list
            input_trunk = xt
            return model.output_strategy.forward(model, 
                                                 data_branch=None, 
                                                 data_trunk=input_trunk, 
                                                 matrices_branch=input_branch, 
                                                 matrices_trunk=None)
        elif self.current_phase == 'branch':
            input_branch = xb
            input_trunk = [aij @ bij for aij, bij in zip(self.T_list, self.R_list)]
            return model.output_strategy.forward(model, 
                                                 data_branch=input_branch, 
                                                 data_trunk=None, 
                                                 matrices_branch=None, 
                                                 matrices_trunk=input_trunk)
        else:
            input_branch = xb
            input_trunk = self.trained_trunk_list
            return model.output_strategy.forward(model, 
                                                 data_branch=input_branch, 
                                                 data_trunk=None, 
                                                 matrices_branch=None, 
                                                 matrices_trunk=input_trunk)
        
    def get_basis_functions(self, **kwargs):
        trunks = self.trained_trunk_list
        basis_functions = torch.stack([net.T for net, _ in zip(trunks, range(len(trunks)))], dim=0)
        return basis_functions

    def update_q_r_t_matrices(self, model, params,  xt):
        with torch.no_grad():
            trunks = model.trunk_networks
            decomposition = params.get('TRUNK_DECOMPOSITION')
            for trunk in trunks:
                phi = trunk(xt)
                if decomposition.lower() == 'qr':
                    logger.info(f"Decomposition using QR factorization...")
                    Q, R = torch.linalg.qr(phi)
                    self.Q_list.append(Q)
                    self.R_list.append(R)

                if decomposition.lower() == 'svd':
                    logger.info(f"Decomposition using SVD...")
                    Q, Sd, Vd = torch.linalg.svd(phi, full_matrices=False)
                    R = torch.diag(Sd) @ Vd
                    self.Q_list.append(Q)
                    self.R_list.append(R)

                T = torch.linalg.inv(R)
                self.T_list.append(torch.linalg.inv(R))

                self.trained_trunk_list.append(Q @ R @ T)

                logger.info(f"Q shape: {Q.shape}, R shape: {R.shape}, T shape: {T.shape}")
                logger.info(f"Reconstructed Phi shape: {(Q @ R @ T).shape}")
                logger.info(f"Q @ R == Phi check: {torch.allclose(Q @ R, phi, atol=1e-6)}")
                logger.info(f"Q matrices: {len(self.Q_list)}\nR matrices: {len(self.R_list)}\nT matrices: {len(self.T_list)}")

            if not self.Q_list or not self.R_list or not self.T_list:
                raise ValueError(
                    f"Trunk decomposition failed. At least one of the matrices wasn't stored.")
            else:
                logger.info(f"Trunk decomposed successfully. \nMoving on to second step...")

    def set_matrices(self, **kwargs):
        self.Q_list = kwargs.get('Q_list')
        self.R_list = kwargs.get('R_list')
        self.T_list = kwargs.get('T_list')

        if not self.Q_list:
            raise ValueError("ERROR: Q matrices couldn't be assigned.")
        if not self.R_list:
            raise ValueError("ERROR: R matrices couldn't be assigned.")
        if not self.T_list:
            raise ValueError("ERROR: T matrices couldn't be assigned.")
        self.trained_trunk_list = [Q @ R @ T for Q, R, T in 
                                    zip(self.Q_list, self.R_list, self.T_list)]
        logger.info(f"Set {len(self.trained_trunk_list)} trained trunk(s) (shaped {(self.trained_trunk_list[0].shape[0], self.trained_trunk_list[0].shape[1])}) for inference.")

    def inference_mode(self):
        self.current_phase = 'final'