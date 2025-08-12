from __future__ import annotations
import torch
import numpy
from copy import deepcopy
from collections import defaultdict
from typing import TYPE_CHECKING
from src.modules.models.deeponet.components.output_handler.config import OutputConfig
from src.modules.models.deeponet.deeponet import DeepONet
from src.modules.models.deeponet.training_strategies.config import TwoStepConfig
from src.modules.models.deeponet.training_strategies.base import TrainingStrategy
from src.modules.models.deeponet.components.component_factory import BranchFactory
from src.modules.models.tools.optimizers.config import OptimizerSpec
from src.modules.models.deeponet.components.trunk.orthonormal_trunk import OrthonormalTrunk
from src.modules.models.deeponet.components.branch.orthonormal_branch import OrthonormalBranch
from src.modules.models.tools.optimizers.optimizer_factory import create_optimizer, create_scheduler
from src.modules.models.deeponet.components.output_handler import SharedBranchHandler, SharedTrunkHandler, SplitOutputsHandler, Phase2Handler
if TYPE_CHECKING:
    from src.modules.models.deeponet.deeponet import DeepONet
    from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig
    from src.modules.models.deeponet.components.component_factory import BranchFactory


class TwoStepStrategy(TrainingStrategy): # TODO: implement share branch decomposition
    """
    A training strategy for DeepONets that proceeds in two distinct phases.
    
    Phase 1: A temporary 'matrix_branch' is used to train a neural 'trunk' network.
    Phase 2: The trained trunk is decomposed to create an orthonormal basis, which is
             then used to train the original neural 'branch' network. This is based on
             the "On the training and generalization of deep operator networks, S.Lee (2024)" paper.
    """
    def __init__(self, config: TwoStepConfig):
        """
        Initializes the TwoStepStrategy with a configuration.
        
        Args:
            config (TwoStepConfig): The configuration object for this strategy.
        """
        super().__init__(config)
        self._phase = 1

    def prepare_components(self, model_config: 'DeepONetConfig'):
        """
        Modifies the model configuration to prepare for Phase 1 training.
        
        This method replaces the user's intended branch and trunk with temporary
        components for the first phase of training.
        
        Args:
            model_config (DeepONetConfig): The DeepONet model configuration.
        """
        self._original_branch_cfg = deepcopy(model_config.branch)
        self._original_trunk_cfg = deepcopy(model_config.trunk)

        # Phase 1 components
        model_config.branch.input_dim = self.config.num_branch_train_samples # type: ignore
        model_config.branch.component_type = "matrix_branch"
        model_config.branch.architecture = "trainable_matrix"
        model_config.trunk.component_type = "neural_trunk"

    def setup_training(self, model: 'DeepONet'):
        """
        Sets up the training environment, including optimizers and schedulers.
        
        Args:
            model (DeepONet): The DeepONet model to be trained.
        """
        # Phase 1: Both components trainable
        model.trunk.requires_grad_(True)
        model.branch.requires_grad_(True)
        model.bias.requires_grad_(True)
        trainable_params = self._get_trainable_parameters(model)
        if not trainable_params:
            raise ValueError("No trainable parameters found in the model.")

        self.trunk_train_schedule = []
        self.branch_train_schedule = []

        for spec in self.config.two_step_optimizer_schedule['trunk_phase']: # type: ignore
            if isinstance(spec, dict):
                spec = OptimizerSpec(**spec)
            trunk_phase_optimizer = create_optimizer(
                spec=spec, params=trainable_params)
            trunk_phase_scheduler = create_scheduler(
                spec, trunk_phase_optimizer)
            self.trunk_train_schedule.append(
                (spec.epochs, trunk_phase_optimizer, trunk_phase_scheduler))

        for spec in self.config.two_step_optimizer_schedule['branch_phase']: # type: ignore
            if isinstance(spec, dict):
                spec = OptimizerSpec(**spec)
            branch_phase_optimizer = create_optimizer(
                spec=spec, params=trainable_params)
            branch_phase_scheduler = create_scheduler(
                spec, branch_phase_optimizer)
            self.branch_train_schedule.append(
                (spec.epochs, branch_phase_optimizer, branch_phase_scheduler))

    def _get_trainable_parameters(self, model: 'DeepONet'):
        """
        Gets all trainable parameters from the model's components.
        
        Args:
            model (DeepONet): The DeepONet model.
            
        Returns:
            list: A list of trainable parameters.
        """
        trainable_params = []
        for name, param in model.trunk.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        for name, param in model.branch.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        for name, param in model.bias.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def get_train_schedule(self) -> list[tuple[torch.optim.optimizer.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]]:
        """
        Returns the appropriate training schedule based on the current phase.
        
        Returns:
            list: The list of (epochs, optimizer, scheduler) tuples for the current phase.
        """
        if self._phase == 1:
            if not hasattr(self, 'trunk_train_schedule'):
                raise ValueError(
                    "Training schedule for trunk not set up. Call setup_training first.")
            return self.trunk_train_schedule
        elif self._phase == 2:
            if not hasattr(self, 'branch_train_schedule'):
                raise ValueError(
                    "Training schedule for branch not set up. Call setup_training first.")
            return self.branch_train_schedule
        else:
            raise RuntimeError("Invalid training phase")

    def execute_phase_transition(self, model: 'DeepONet', all_branch_indices: torch.Tensor, full_trunk_batch: torch.Tensor, full_outputs_batch: torch.Tensor):
        """
        Performs the transition from Phase 1 to Phase 2.
        
        This involves decomposing the trained trunk, creating a new orthonormal trunk,
        and replacing the temporary branch with the original one from the config.
        
        Args:
            model (DeepONet): The DeepONet model.
            all_branch_indices (torch.Tensor): Indices for all branch inputs.
            full_trunk_batch (torch.Tensor): A full batch of trunk inputs.
            full_outputs_batch (torch.Tensor): A full batch of corresponding outputs.
        """
        if not isinstance(self.config, TwoStepConfig):
            raise TypeError("TwoStepStrategy requires TwoStepConfig")

        if self._phase != 1:
            raise RuntimeError("Phase transition from invalid state")

        R, T = self._decompose_trunk(
            model=model,
            trunk=model.trunk, 
            full_trunk_batch=full_trunk_batch
        )

        self._check_RT_invertibilty(model, R, T)

        
        self.R = R
        self.A = model.branch
        self.all_branch_indices = all_branch_indices

        with torch.no_grad():
            self.A_full = self.A(self.all_branch_indices.flatten())

        new_trunk_config = deepcopy(self._original_trunk_cfg)
        new_trunk_config.component_type = "orthonormal_trunk"
        new_trunk_config.architecture = "pretrained"
        new_trunk_config.inner_config = deepcopy(
            self._original_trunk_cfg)  # Preserve original

        self.final_branch_config = deepcopy(self._original_branch_cfg)
        new_branch_config = deepcopy(self._original_branch_cfg)
        new_branch_config.component_type = "orthonormal_branch"
        new_branch_config.architecture = "pretrained"
        new_branch_config.inner_config = deepcopy(
            self._original_branch_cfg)  # Preserve original
        
        self.final_trunk_config = new_trunk_config
        self.final_branch_config = new_branch_config

        if isinstance(model.output_handler, (SplitOutputsHandler)):
            if self.final_branch_config.output_dim is not None:
                self.final_branch_config.output_dim *= model.output_handler.num_channels
            if self.final_branch_config.inner_config and self.final_branch_config.inner_config.output_dim is not None:
                self.final_branch_config.inner_config.output_dim *= model.output_handler.num_channels
            if self.final_trunk_config.output_dim is not None:
                self.final_trunk_config.output_dim *= model.output_handler.num_channels
            if self.final_trunk_config.inner_config and self.final_trunk_config.inner_config.output_dim is not None:
                self.final_trunk_config.inner_config.output_dim *= model.output_handler.num_channels
        elif isinstance(model.output_handler, (SharedBranchHandler)):
            if self.final_trunk_config.output_dim is not None:
                self.final_trunk_config.output_dim *= model.output_handler.num_channels
            if self.final_trunk_config.inner_config and self.final_trunk_config.inner_config.output_dim is not None:
                self.final_trunk_config.inner_config.output_dim *= model.output_handler.num_channels
        else:
            if self.final_branch_config.output_dim is not None:
                self.final_branch_config.output_dim *= model.output_handler.num_channels
            if self.final_branch_config.inner_config and self.final_branch_config.inner_config.output_dim is not None:
                self.final_branch_config.inner_config.output_dim *= model.output_handler.num_channels

        self.output_handler = model.output_handler

        if self._original_branch_cfg is None:
            raise RuntimeError("Missing original component configs.")

        if self.final_branch_config.inner_config is None:
            raise RuntimeError("Final branch config must have an inner config.")
        new_inner_branch = BranchFactory.build(self.final_branch_config.inner_config)
        
        is_shared_branch = isinstance(model.output_handler, SharedBranchHandler)
        is_shared_trunk = isinstance(model.output_handler, SharedTrunkHandler)
        num_channels = model.output_handler.num_channels
        handler_cfg = OutputConfig(handler_type='two_step_final', num_channels=num_channels)

        final_branch = OrthonormalBranch(new_inner_branch, R, num_channels, is_shared_branch)
        final_trunk = OrthonormalTrunk(model.trunk, T, num_channels, is_shared_trunk)
        final_handler = Phase2Handler(config=handler_cfg)
        
        model.trunk = final_trunk
        model.branch = final_branch
        model.output_handler = final_handler

        model.trunk.requires_grad_(False)
        model.branch.requires_grad_(True)

        model.trunk.eval()

        self._check_orthonormality(model, full_trunk_batch, T)
        self._check_target_reconstruction(model, full_trunk_batch, full_outputs_batch, T)

        self._update_optimizer_parameters(model)

        self._phase = 2

    def _check_RT_invertibilty(self, model: 'DeepONet', R: torch.Tensor, T: torch.Tensor):
        if isinstance(model.output_handler, SplitOutputsHandler):
            n_channels = model.output_handler.num_channels
            for i in range(n_channels):
                rows = slice(i * R.shape[0] // n_channels, (i + 1) * R.shape[0] // n_channels)
                R_i = R[rows, rows]
                T_i = T[rows, rows]

                should_be_I = R_i @ T_i

                # if not torch.allclose(should_be_I, torch.eye(R_i.shape[0]), atol=1e-4):
                #     raise RuntimeError(f"Decomposition did not produce identity matrix for channel {i}.")

        # else:
        #     should_be_I = R @ T
        #     # if not torch.allclose(should_be_I, torch.eye(R.shape[0]), atol=1e-4):
        #     #     raise RuntimeError("Decomposition did not produce identity matrix.")
        

    def _check_orthonormality(self, model: 'DeepONet', full_trunk_batch: torch.Tensor, T: torch.Tensor):
        """Checks if the trunk outputs are orthonormal after decomposition."""
        with torch.no_grad():
            old_trunk = model.trunk.trunk           # unwrap the base trunk
            phi      = old_trunk(full_trunk_batch)  # (T , C·P or P)
            C = model.output_handler.num_channels

            if isinstance(model.output_handler, SplitOutputsHandler): # need to refactor this for n channels
                pass
                # CtimesP = phi.shape[1]
                # P = CtimesP // C
                # I =  torch.eye(P, device=phi.device)

                # # 1)  columns = phi @ T1
                # Z1 = phi @ T[:, : P]                      # should be orthonormal
                # err1 = torch.linalg.norm(Z1.T @ Z1 - I) / torch.linalg.norm(I)

                # # 2)  columns = phi @ T2
                # print(phi.shape, T.shape, T[:, : P].shape, T[:, P :].shape)
                # quit()
                # Z2 = phi @ T[:, P :]
                # err2 = torch.linalg.norm(Z2.T @ Z2 - I) / torch.linalg.norm(I)

                # print('‖I - (phi T)ᵀ(phi T)‖₂ / ‖I‖ =', f"{err1.item():.6%}")
                # print('‖I - (phi Tᵀ)ᵀ(phi Tᵀ)‖₂ / ‖I‖ =', f"{err2.item():.6%}")
                # if 100 * err1 > 1 or 100 * err2 > 1:
                #     raise RuntimeError("Trunk outputs are not orthonormal after decomposition.")
        
    def _check_target_reconstruction(self, model: 'DeepONet', full_trunk_batch: torch.Tensor, full_outputs_batch: torch.Tensor, T: torch.Tensor):
        pass
        # with torch.no_grad():
        #     baseline_Y  = full_outputs_batch  # whatever you used as ground‑truth in phase 1
        #     C = model.output_handler.num_channels
        #     A = self.A_full.reshape(full_outputs_batch.shape[0], C, -1)
        #     coeff = self._matmul_blockwise(A, self.R.T) # (B, C, P)
        #     trunk_out = model.trunk(full_trunk_batch)          # (T, C·P)
        #     raw_trunk_out =   model.trunk.trunk(full_trunk_batch)

        #     if isinstance(model.output_handler, SplitOutputsHandler):
        #         # 1) coefficients in the orthonormal basis  C = A Rᵀ
        #         B, C, P = coeff.shape

        #         # 2) trunk evaluated on every location, already orthonormal

        #         print("coeff shape:", coeff.shape)         # (B, C, P)
        #         print("trunk_out shape:", trunk_out.shape) # (T, C, P)

        #         # Check norms channel-wise
        #         print("||coeff|| per channel:", coeff.norm(dim=(0,2)))
        #         print("||trunk_out|| per channel:", trunk_out.reshape(-1, C, P).norm(dim=(0,2)))
        #         print("||baseline_Y||:", baseline_Y.norm())

        #         T_loc = trunk_out.shape[0]
        #         trunk_out = trunk_out.view(T_loc, C, P)            # (T, C, P)
        #         raw_trunk_out = raw_trunk_out.view(T_loc, C, P)            # (T, C, P)

        #         # 3) reproduce the model’s combine() logic
        #         raw_Y_recon = model.rescaler(torch.einsum('bcp,tcp->btc', A, raw_trunk_out) + model.bias.bias) # (B, T, C)
        #         Y_recon = model.rescaler(torch.einsum('bcp,tcp->btc', coeff, trunk_out) + model.bias.bias)  # (B, T, C)

        #         # 4) compare with ground‑truth used in phase 1
        #         rel_err = (Y_recon - baseline_Y).norm(dim=(0, 1)) / baseline_Y.norm(dim=(0, 1)).detach().numpy()
        #         for ch in range(C):
        #             print(f'ϵ_rel_{ch}(Y_recon) = {rel_err[ch]:.3e}')
                
        #         dot_err = (Y_recon - raw_Y_recon).norm() / raw_Y_recon.norm()
        #         print(f'ϵ_rel(dot_product) = {dot_err:.3e}')

        #     else:
        #         trunk_out = model.trunk(full_trunk_batch)          # (T, C·P)
        #         print("coeff shape:", coeff.shape)         # (B, C, P)
        #         print("trunk_out shape:", trunk_out.shape) # (T, P)

        #         # Check norms channel-wise
        #         print("||coeff|| per channel:", coeff.norm(dim=(0,2)))
        #         print("||trunk_out||:", trunk_out.norm(dim=(0,1)))
        #         print("||baseline_Y||:", baseline_Y.norm())

        #         T_loc = trunk_out.shape[0]

        #         # 3) reproduce the model’s combine() logic
        #         raw_Y_recon = model.rescaler(torch.einsum('bcp,tp->btc', A, raw_trunk_out) + model.bias.bias)  # (B, T, C)
        #         Y_recon = model.rescaler(torch.einsum('bcp,tp->btc', coeff, trunk_out) + model.bias.bias)  # (B, T, C)
        #         rel_err = (Y_recon - baseline_Y).norm(dim=(0, 1)) / baseline_Y.norm(dim=(0, 1)).detach().numpy()
        #         for ch, r_e in enumerate(rel_err):
        #             print(f'ϵ_rel_{ch}(Y_recon) = {r_e:.3e}')

        #         dot_err = (Y_recon - raw_Y_recon).norm() / raw_Y_recon.norm()
        #         print(f'ϵ_rel(dot_product) = {dot_err:.3e}')
            


    def _update_optimizer_parameters(self, model: 'DeepONet'):
        """
        Updates the optimizers for Phase 2 to only consider the new branch parameters.
        
        Args:
            model (DeepONet): The DeepONet model.
        """
        trainable_params = []
        for param in model.branch.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        for _, optimizer, _ in self.branch_train_schedule:
            optimizer.param_groups.clear()
            optimizer.add_param_group({'params': trainable_params})

        for _, optimizer, _ in self.branch_train_schedule:
            optimizer.state = defaultdict(dict)

    def validation_enabled(self) -> bool:
        """
        Checks if validation is enabled. It is not enabled by default for this strategy.
        """
        return False

    def should_transition_phase(self, current_phase: int, current_epoch: int) -> bool:
        """
        Determines if a phase transition from Phase 1 to Phase 2 is needed.
        
        The transition happens after all epochs for the trunk phase are complete.
        
        Args:
            current_phase (int): The current training phase (1 or 2).
            current_epoch (int): The current epoch within the current phase.
            
        Returns:
            bool: True if a transition should occur, False otherwise.
        """
        if current_phase == 1:
            # Transition after completing trunk phase
            trunk_epochs = sum(epochs for epochs, _,
                               _ in self.trunk_train_schedule)
            return current_epoch >= trunk_epochs
        return False

    def compute_loss(self, 
                     model: DeepONet,
                     x_branch: torch.Tensor,
                     x_trunk: torch.Tensor,
                     y_true: torch.Tensor,
                     indices: tuple[numpy.ndarray, ...]) -> tuple[torch.Tensor, ...]:
        """
        Computes the loss for the given model and data, adjusting based on the phase.
        
        Args:
            model (DeepONet): The DeepONet model.
            x_branch (torch.Tensor): Branch inputs.
            x_trunk (torch.Tensor): Trunk inputs.
            y_true (torch.Tensor): Ground truth outputs.
            indices (tuple): Indices for branch inputs.
            
        Returns:
            tuple: A tuple of (y_pred, loss).
        """

        if self._phase == 1:
            y_pred = model(indices[0], x_trunk)
            loss = self.loss(y_pred, y_true)
        elif self._phase == 2:
            synthetic_targets = self.compute_synthetic_targets(model=model, branch_indices=indices[0])
            A_pred_raw = model.branch(x_branch)
            
            if isinstance(model.output_handler, SharedBranchHandler):
                A_pred_flat = A_pred_raw.unsqueeze(1)
                A_pred = self._matmul_blockwise(
                                    A_flat=A_pred_flat,
                                    R_T=self.R.T,
                                    n_channels=model.output_handler.num_channels
                                )
                loss = self.loss(A_pred, synthetic_targets)

            else:
                A_pred = A_pred_raw.reshape(synthetic_targets.shape)
                loss = self.loss(A_pred, synthetic_targets)
            trunk_out = model.trunk(x_trunk)
            branch_out = A_pred
            y_pred = model.rescaler(model.output_handler.combine(branch_out, trunk_out))
        else:
            raise RuntimeError("Invalid training phase")
        return y_pred, loss

    def calculate_metrics(self,
                          model: 'DeepONet',
                          y_true: torch.Tensor,
                          y_pred: torch.Tensor,
                          loss: float,
                          train: bool,
                          branch_indices: numpy.ndarray,
                          label_map: list[str] | None = None) -> dict[str, float]:
        """Combines base and strategy-specific metrics"""
        if self._phase == 1:
            metrics = self.base_metrics(y_true, y_pred, loss, label_map)
        else:
            metrics = {'loss': loss}
        metrics.update(self.strategy_specific_metrics(
            model=model,
            y_true=y_true, y_pred=y_pred, branch_indices=branch_indices, label_map=label_map))
        return metrics

    def strategy_specific_metrics(self, model: 'DeepONet', y_true: torch.Tensor, y_pred: torch.Tensor, branch_indices: numpy.ndarray, label_map: list[str] | None = None) -> dict[str, float]:
        """
        Calculates strategy-specific metrics, such as relative error, for each phase.
        
        Args:
            model (torch.nn.Module): The model.
            y_true (torch.Tensor): Ground truth outputs.
            y_pred (torch.Tensor): Predicted outputs.
            branch_indices (numpy.ndarray): Indices for branch inputs.
            label_map (list[str]): Optional list of labels for outputs.
            
        Returns:
            dict: A dictionary of the calculated metrics.
        """
        if self._phase == 1:
            relative_error = self.error_metric(
                y_true - y_pred) / self.error_metric(y_true)
        elif self._phase == 2:
            error_norm = self.error_metric(
                y_pred - y_true)
            true_norm = self.error_metric(y_true)
            relative_error = error_norm / (true_norm + 1e-8)
        else:
            raise RuntimeError("Invalid training phase")

        if relative_error.ndim > 0:
            if label_map is not None:
                strategy_metric = {
                    **{f'Error_{label_map[i]}': e.item() for i, e in enumerate(relative_error.detach())}
                }
            else:
                strategy_metric = {
                    **{f'Error_{i}': e.item() for i, e in enumerate(relative_error.detach())}
                }
        else:
            strategy_metric = {f'Error': relative_error.item()}
        return strategy_metric

    def _matmul_blockwise(self,
                          A_flat: torch.Tensor,   # (B, C, P) or (B, 1, C)
                          R_T: torch.Tensor,   # (C·P, C·P) or (P, P)
                          n_channels: int
                        ) -> torch.Tensor:
        """
        Args:
            y_true (torch.Tensor): Ground truth outputs.
            y_pred (torch.Tensor): Predicted outputs.
            branch_indices (int): Size of operator output vector.

        Returns:
            coeffs (torch.Tensor): coefs = M = A · (Rᵀ). It occurs block‑wise on Rᵀ if trunk is split.
        Block multiplication will depend of output handling (i = 1, ... , C):
            1) If 'split_outputs', we have M_i = A_i @ block_i(Rᵀ): 
                - D = C
                - (B, C, P) (block-wise product) (C·P, C·P) -> (B, C, P)
            2) If 'shared_trunk', we have M_i = A_i @ (Rᵀ):
                - D = C
                - (B, C, P)        @             (P, P) -> (B, C, P)
            3) If 'shared_branch', we have M_i = A @ block_i(Rᵀ)
                - D = 1
                - (B, 1, P) (block-wise product) (C·P, C·P) -> (B, C, P)
        """
        C = n_channels
        B, D, P = A_flat.shape
        coeff = torch.empty((B, C, P), device=A_flat.device, dtype=A_flat.dtype)

        for c in range(C):
            rows = slice(c * P, (c + 1) * P)

            if R_T.shape[0] == C * P:
                R_c = R_T[rows, rows]  # Always (P, P) block
                Rc = R_c
            else:
                Rc = R_T
            if D == C:
                Ac = A_flat[:, c, :]
                coeff[:, c, :] = Ac @ Rc
            else:
                A = A_flat[:, 0]
                coeff[:, c, :] = A @ Rc

        return coeff # (B, C, P)

    def _broadcast_coeffs(self, coeff_pred, coeff_pretrained) -> torch.Tensor:
        pred_shape = coeff_pred.shape # Either (B, P) or (B, C·P)
        true_shape = coeff_pretrained.shape # Always (B, P, C)
        
        if pred_shape[-1] != true_shape[-2]:
            coeff_pred_reshaped = coeff_pred.reshape(true_shape)
        else:
            coeff_pred_reshaped = coeff_pred[:, :, None]
        return coeff_pred_reshaped
                
    def compute_synthetic_targets(self, model: 'DeepONet', branch_indices: numpy.ndarray) -> torch.Tensor:
        """Computes synthetic targets using the decomposed trunk and branch"""
        if self.A is None or self.R is None:
            raise RuntimeError(
                "A and R matrices must be set before computing synthetic targets.")
        
        def _handle_A_reshape(model: 'DeepONet', A: torch.Tensor) -> torch.Tensor:
            B, D = A.shape
            if model.branch.is_shared_branch and not model.trunk.is_shared_trunk : # Here, D = P (always)
                P = D
                C = 1
            else:   # D = C * P where P is the embedding size
                C = model.output_handler.num_channels
                P = D // C
            return A.view(B, C, P)

        indices = branch_indices.flatten()
        A_batch = self.A_full[indices]
        A_flat = _handle_A_reshape(model, A_batch)
        
        coeff = self._matmul_blockwise(
            A_flat=A_flat, 
            R_T=self.R.T, 
            n_channels=model.output_handler.num_channels
        )
        return coeff

    def get_optimizer_scheduler(self):
        return self.config.optimizer_scheduler  # type: ignore

    def get_phases(self) -> list[str]:
        """Return phase names (e.g., ['phase1', 'phase2'])"""
        return ["trunk_phase", "branch_phase"]

    def apply_gradient_constraints(self, model: DeepONet):
        """Optional gradient clipping/normalization"""
        pass

    def _decompose_trunk(self, model: 'DeepONet', trunk: torch.nn.Module, full_trunk_batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """SVD decomposition of final trunk layer.
        Returns (R, T) with shapes            (D, D)  and  (D, D)
        where D = C·P  for split‑channel handlers,  D = P otherwise.
        """

        was_training = trunk.training
        trunk.eval()

        with torch.no_grad():
            phi = trunk(full_trunk_batch)
        if was_training:
            trunk.train()

        D = phi.shape[1]

        handler = model.output_handler
        split = isinstance(handler, SplitOutputsHandler)
        shared_branch = isinstance(handler, SharedBranchHandler)
        C = getattr(handler, 'num_channels')
        if split or shared_branch:
            if D % C != 0:
                raise ValueError(
                    f"Trunk outputs {D} columns but handler expects "
                    f"{C} channels -> columns must be divisible by channels."
                )
            P = D // C
        else:
            P = D

        R_blocks, T_blocks = [],[]

        def _svd_block(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            _, S, Vh = torch.linalg.svd(mat, full_matrices=True)
            R = torch.diag(S) @ Vh
            T = Vh.T @ torch.diag(1.0 / S)
            return R, T
        
        def _qr_block(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            _, R = torch.linalg.qr(mat)
            T = torch.linalg.inv(R)
            return R, T

        use_qr = (self.config.decomposition_type.lower() == "qr") # type: ignore

        if split or shared_branch:
            for c in range(C):
                cols = slice(c * P, (c + 1) * P)
                phi_ch = phi[:, cols]
                if use_qr:
                    R_ch, T_ch = _qr_block(phi_ch)
                else:
                    R_ch, T_ch = _svd_block(phi_ch)
                R_blocks.append(R_ch)
                T_blocks.append(T_ch)

            R = torch.block_diag(*R_blocks) # (C·P, C·P)
            T = torch.block_diag(*T_blocks) 
        else:
            R, T = (_qr_block(phi) if use_qr else _svd_block(phi)) # (P, P)

        if was_training:
            trunk.train()

        return R, T
