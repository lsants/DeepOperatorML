import torch
from abc import ABC, abstractmethod
from ..loss_complex import loss_complex

class TrainingStrategy(ABC):
    @abstractmethod
    def prepare_training(self, model):
        """
        Prepares model for training (e.g. initializes/freezes trainable parameters)

        Args:
            model (DeepONet): Model instance.
        """
        pass

    @abstractmethod
    def update_training_phase(self, model, phase):
        """
        Updated the training phase of the model.

        Args:
            model (DeepONet): The model's instance.
            phase (str): The current training phase.
        """
        pass

    def forward(self, model, xb, xt):
        """
        Optional forward pass override. If implemented, this method will be used instead of the OutputHandlingStrategy's forward.

        Args:
            model (DeepONet): The model instance.
            xb (list or torch.Tensor): Inputs to the branch networks.
            xt (list or torch.Tensor): Inputs to the trunk networks.

        Returns:
            tuple: Outputs as determined by the training strategy.
        """
        return model.output_strategy.forward(model, xb, xt)
    
    def get_trunk_output(self, model, i, xt_i):
        """
        Retrieves the trunk output for the i-th output. Can be overridden by training strategies.

        Args:
            model (DeepONet): The model instance.
            i (int): Index of the output.
            xt_i (torch.Tensor): Input to the trunk network for the i-th output.

        Returns:
            torch.Tensor: Trunk output for the i-th output.
        """
        if hasattr(model, 'trunk_networks'):
            return model.trunk_networks[i](xt_i)
        else:
            return model.trunk_network(xt_i)
        
    def get_branch_output(self, model, i, xb_i):
        """
        Retrieves the branch output for the i-th output. Can be overridden by training strategies.

        Args:
            model (DeepONet): The model instance.
            i (int): Index of the output.
            xb_i (torch.Tensor): Input to the branch network for the i-th output.

        Returns:
            torch.Tensor: Branch output for the i-th output.
        """
        return model.branch_networks[i](xb_i)
    
class StandardTrainingStrategy(TrainingStrategy):
    def prepare_training(self, model):
        """
        Prepares the model for standard training by ensuring all networks are trainable.

        Args:
            model (DeepONet): The model instance.
        """
        for param in model.trunk_network.parameters():
            param.requires_grad = True
        for branch in model.branch_networks:
            for param in branch.parameters():
                param.requires_grad = True
    
    def update_training_phase(self, model, phase):
        """
        Standard training does not require phase updates.

        Args:
            model (DeepONet): The model instance.
            phase (str): Not used in standard training.
        """
        pass

class TwoStepTrainingStrategy(TrainingStrategy):
    def __init__(self, A_dim, n_outputs):
        """
        Initializes the Two-Step training strategy.

        Args:
            A_dim (tuple): Size of each matrix A.
            n_outputs (int): Number of outputs.
        """
        assert len(A_dim) == n_outputs, "A_dim must match the number of outputs"
        self.A_list = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(A_dim))
            for _ in range(n_outputs)
        ])

        for A in self.A_list:
            torch.nn.init.xavier_uniform_(A)

        self.Q = None
        self.R = None
        self.T = None
        self.current_phase = 'trunk'

    def prepare_training(self, model):
        """
        Prepares the model for two-step training by freezing branch networks during trunk training.

        Args:
            model (DeepONet): The model instance.
        """
        self.current_phase = 'trunk'
        for branch in model.branch_networks:
            for param in branch.parameters():
                param.requires_grad = False
        if hasattr(model, 'trunk_networks'):
            for trunk in model.trunk_networks:
                for param in trunk.parameters():
                    param.requires_grad = True
        else:
            for param in model.trunk_network.parameters():
                param.requires_grad = True
    
    def update_training_phase(self, model, phase):
        """
        Updates the training phase and adjusts parameter freezing accordingly.

        Args:
            model (DeepONet): The model instance.
            phase (str): The new training phase ('trunk', 'branch', 'final').
        """
        self.current_phase = phase
        if phase == 'trunk':
            if hasattr(model, 'trunk_networks'):
                for trunk in model.trunk_networks:
                    for param in trunk.parameters():
                        param.requires_grad = True
            else:
                for param in model.trunk_network.parameters():
                    param.requires_grad = True

            for branch in model.branch_networks:
                for param in branch.parameters():
                    param.requires_grad = False
        elif phase == 'branch':
            if hasattr(model, 'trunk_networks'):
                for trunk in model.trunk_networks:
                    for param in trunk.parameters():
                        param.requires_grad = False
            else:
                for param in model.trunk_network.parameters():
                    param.requires_grad = False

            for branch in model.branch_networks:
                for param in branch.parameters():
                    param.requires_grad = True

            self.update_q_r_t_matrices(model)

        elif phase == 'final':
            if hasattr(model, 'trunk_networks'):
                for trunk in model.trunk_networks:
                    for param in trunk.parameters():
                        param.requires_grad = False
            else:
                for param in model.trunk_network.parameters():
                    param.requires_grad = False

            for branch in model.branch_networks:
                for param in branch.parameters():
                    param.requires_grad = False

    def compute_loss(self, outputs, batch, model):
        """
        Computes the loss based on the current training phase.

        Args:
            outputs (tuple): Outputs from the model.
            batch (dict): Batch data.
            model (DeepONet): The model instance.

        Returns:
            torch.Tensor: Computed loss.
        """
        if self.current_phase == 'trunk':
            pred_real, pred_imag = outputs
            target_real = batch['g_u_real']
            target_imag = batch['g_u_imag']
            loss = loss_complex(target_real, target_imag, pred_real, pred_imag)
        elif self.current_phase == 'branch':
            preds = outputs
            targets = []
            for i in range(self.n_outputs):
                target = self.R_list[i] @ self.A_list[i]
                targets.append(target)
            loss = loss(*targets, *preds)
        elif self.current_phase == 'final':
            targets = batch['targets']
            loss = loss(*targets, *outputs)

        else:
            raise ValueError(f"Unknown training phase: {self.current_phase}")
        return loss

    def compute_errors(self, outputs, batch, model):
        """
        Computes errors based on the current training phase.

        Args:
            outputs (tuple): Outputs from the model.
            batch (dict): Batch data.
            model (DeepONet): The model instance.

        Returns:
            dict: Dictionary containing error metrics.
        """
        if self.current_phase == 'trunk':
            pred_real, pred_imag = outputs
            target_real = batch['g_u_real']
            target_imag = batch['g_u_imag']

            error_real = torch.norm(target_real - pred_real, p=2).item()
            error_imag = torch.norm(target_imag - pred_imag, p=2).item()

        elif self.current_phase == 'branch':
            # Compute errors on branch outputs
            error_real = 0.0
            error_imag = 0.0
            for i in range(self.n_outputs):
                branch_output = outputs[i]
                A_target = self.A_list[i].to(branch_output.device)
                error_real += torch.norm(A_target - branch_output, p=2).item()
                error_imag += torch.norm(A_target - branch_output, p=2).item()
        elif self.current_phase == 'final':
            # Compute errors on final outputs
            error_real = 0.0
            error_imag = 0.0
            for i in range(self.n_outputs):
                pred = outputs[i]
                # Assuming you have target_final_real and target_final_imag in batch
                target_final_real = batch['target_final_real'][i]
                target_final_imag = batch['target_final_imag'][i]
                error_real += torch.norm(target_final_real - pred.real, p=2).item()
                error_imag += torch.norm(target_final_imag - pred.imag, p=2).item()
        else:
            raise ValueError(f"Unknown training phase: {self.current_phase}")
        return {'real': error_real, 'imag': error_imag}

    def can_validate(self):
        """
        Two-Step training doesn't supports validation.

        Returns:
            bool: True
        """
        return False

    def before_epoch(self, epoch, model, params):
        """
        Handles any logic before each epoch.

        Args:
            epoch (int): Current epoch number.
            model (DeepONet): The model instance.
            params (dict): Training parameters.
        """
        pass

    def after_epoch(self, epoch, model, params):
        """
        Handles any logic after each epoch, such as phase transitions.

        Args:
            epoch (int): Current epoch number.
            model (DeepONet): The model instance.
            params (dict): Training parameters.
        """
        if self.current_phase == 'trunk' and epoch + 1 >= params['TRUNK_TRAIN_EPOCHS']:
            self.current_phase = 'branch'
            self.update_training_phase(model, 'branch')
            print(f"Transitioned to 'branch' phase at epoch {epoch + 1}")
        elif self.current_phase == 'branch' and epoch + 1 >= (params['TRUNK_TRAIN_EPOCHS'] + params['BRANCH_TRAIN_EPOCHS']):
            self.current_phase = 'final'
            self.update_training_phase(model, 'final')
            print(f"Transitioned to 'final' phase at epoch {epoch + 1}")

    def get_optimizers(self, model, params):
        """
        Defines and returns the optimizer.

        Args:
            model (DeepONet): The model instance.
            params (dict): Training parameters.

        Returns:
            dict: Dictionary containing the optimizer.
        """
        if self.current_phase == 'trunk':
            trunk_params = []
            if hasattr(model, 'trunk_networks'):
                for trunk in model.trunk_networks:
                    trunk_params += list(trunk.parameters())
            else:
                trunk_params = list(model.trunk_network.parameters())
            optimizer = torch.optim.Adam(trunk_params, lr=params['LEARNING_RATE'], weight_decay=params['L2_REGULARIZATION'])
        elif self.current_phase == 'branch':
            branch_params = []
            for branch in model.branch_networks:
                branch_params += list(branch.parameters())
            optimizer = torch.optim.Adam(branch_params, lr=params['LEARNING_RATE'], weight_decay=params['L2_REGULARIZATION'])
        elif self.current_phase == 'final':
            optimizer = None
        else:
            raise ValueError(f"Unknown training phase: {self.current_phase}")

        return {'optimizer': optimizer} if optimizer else {}

    def get_schedulers(self, optimizers, params):
        """
        Defines and returns the scheduler.

        Args:
            optimizers (dict): Dictionary containing the optimizer.
            params (dict): Training parameters.

        Returns:
            dict: Dictionary containing the scheduler.
        """
        if 'optimizer' in optimizers and optimizers['optimizer']:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizers['optimizer'],
                step_size=params['SCHEDULER_STEP_SIZE'],
                gamma=params['SCHEDULER_GAMMA']
            )
            return {'scheduler': scheduler}
        else:
            return {}

    def zero_grad(self, optimizers):
        """
        Zeroes the gradients for all optimizers.

        Args:
            optimizers (dict): Dictionary containing the optimizer.
        """
        if 'optimizer' in optimizers and optimizers['optimizer']:
            optimizers['optimizer'].zero_grad()

    def step(self, optimizers):
        """
        Performs an optimizer step.

        Args:
            optimizers (dict): Dictionary containing the optimizer.
        """
        if 'optimizer' in optimizers and optimizers['optimizer']:
            optimizers['optimizer'].step()

    def step_schedulers(self, schedulers):
        """
        Steps the schedulers.

        Args:
            schedulers (dict): Dictionary containing the scheduler.
        """
        if 'scheduler' in schedulers and schedulers['scheduler']:
            schedulers['scheduler'].step()

        
    def forward(self, model, xb, xt):
        """
        Forward pass using two-step training strategy.

        Args:
            model (DeepONet): The model instance.
            xb (list of torch.Tensor): Inputs to the branch networks.
            xt (list of torch.Tensor, optional): Inputs to the trunk networks.

        Returns:
            tuple: Outputs as determined by the OutputHandlingStrategy.
        """
        return model.output_strategy.forward(model, xb, xt)
        
    def get_trunk_output(self, model, i, xt_i):
        """
        Retrieves the trunk output for the i-th output based on the current phase.
        In the first step, the output is always the trunk network's output.
        In the second step, the trunk network is not used (the identity matrix is returned).
        After training, the output is the product of matrices Q, R and T.

        Args:
            model (DeepONet): The model instance.
            i (int): Index of the output.
            xt_i (torch.Tensor): Input to the trunk network for the i-th output.

        Returns:
            torch.Tensor: Trunk output for the i-th output.
        """
        if self.current_phase == 'trunk':
            if hasattr(model, 'trunk_networks'):
                return model.trunk_networks[i](xt_i)
            else:
                return model.trunk_network(xt_i)
        elif self.current_phase == 'branch':
            R_i = self.R_list[i]
            N = len(R_i)
            return torch.eye(N)
        elif self.current_phase == 'final':
            Q_i = self.Q_list[i]
            R_i = self.R_list[i]
            T_i = self.T_list[i]
            return Q_i @ R_i @ T_i
        else:
            raise ValueError(f"Unknown training phase: {self.current_phase}")
        
    def get_branch_output(self, model, i, xb_i):
        """
        Retrieves the branch output for the i-th output based on the current phase.

        Args:
            model (DeepONet): The model instance.
            i (int): Index of the output.
            xb_i (torch.Tensor): Input to the branch network for the i-th output.

        Returns:
            torch.Tensor: Branch output for the i-th output.
        """
        if self.current_phase == 'trunk':
            A_i = self.A_list[i]
            return A_i
        elif self.current_phase == 'branch':
            return model.branch_networks[i](xb_i)
        elif self.current_phase == 'final':
            return model.branch_networks[i](xb_i)
        else:
            raise ValueError(f"Unknown training phase: {self.current_phase}")
        
class PODTrainingStrategy(TrainingStrategy):
    def __init__(self, pod_basis, mean_functions):
        """
        Initializes the POD training strategy.

        Args:
            pod_basis (torch.Tensor): Precomputed POD basis matrices.
                                       Shape: (n_outputs, num_modes, features)
            mean_functions (torch.Tensor): Precomputed mean functions.
                                           Shape: (n_outputs, features)
        """
        self.pod_basis = pod_basis
        self.mean_functions = mean_functions

    def prepare_training(self, model):
        """
        Prepares the model for POD-based training by integrating POD basis and freezing the trunk networks.

        Args:
            model (DeepONet): The model instance.
        """
        for i in range(model.n_outputs):
            model.register_buffer(f'pod_basis_{i}', self.pod_basis[i])
            model.register_buffer(f'mean_functions_{i}', self.mean_functions[i])
        
        for trunk in model.trunk_networks:
            for param in trunk.parameters():
                param.requires_grad = False

    def update_training_phase(self, model, phase):
        """
        Not used here.

        Args:
            model (DeepONet): The model instance.
            phase (str): Not used in POD training.
        """
        pass

    def get_trunk_output(self, model, i, xt_i):
        """
        Overrides the trunk output to use pod_basis instead of trunk_network.

        Args:
            model (DeepONet): The model instance.
            i (int): Index of the output.
            xt_i (torch.Tensor): Input to the trunk network for the i-th output (unused).

        Returns:
            torch.Tensor: Trunk output for the i-th output, which is pod_basis.
        """
        return getattr(model, f'pod_basis_{i}')
    
    def get_branch_output(self, model, i, xb_i):
        """
        Optionally, modify the branch output if needed. For POD, branches are used as is.

        Args:
            model (DeepONet): The model instance.
            i (int): Index of the output.
            xb_i (torch.Tensor): Input to the branch network for the i-th output.

        Returns:
            torch.Tensor: Branch output for the i-th output.
        """
        return model.branch_networks[i](xb_i)