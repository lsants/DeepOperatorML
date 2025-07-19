from __future__ import annotations
import torch
from copy import deepcopy
from collections import defaultdict
from ...model.deeponet import DeepONet
from .base import TrainingStrategy
from typing import TYPE_CHECKING
from .config import TwoStepConfig
from ..components.component_factory import BranchFactory
from ..optimization.optimizers.config import OptimizerSpec
from ..components.trunk.orthonormal_trunk import OrthonormalTrunk
from ..optimization.optimizers.optimizer_factory import create_optimizer, create_scheduler
from ..components.output_handler import SharedTrunkHandler, SplitOutputsHandler
if TYPE_CHECKING:
    from ..deeponet import DeepONet
    from ..config import ModelConfig
    from ..components.component_factory import BranchFactory, TrunkFactory


class TwoStepStrategy(TrainingStrategy):
    def __init__(self, config: TwoStepConfig):
        super().__init__(config)
        self._phase = 1

    def prepare_components(self, model_config: 'ModelConfig'):
        self._original_branch_cfg = deepcopy(model_config.branch)
        self._original_trunk_cfg = deepcopy(model_config.trunk)

        # Phase 1 components
        model_config.branch.component_type = "matrix_branch"
        model_config.branch.architecture = "trainable_matrix"
        model_config.trunk.component_type = "neural_trunk"

    def setup_training(self, model: 'DeepONet'):
        # Phase 1: Both components trainable
        model.trunk.requires_grad_(True)
        model.branch.requires_grad_(True)
        model.bias.requires_grad_(True)
        trainable_params = self._get_trainable_parameters(model)
        if not trainable_params:
            raise ValueError("No trainable parameters found in the model.")

        self.trunk_train_schedule = []
        self.branch_train_schedule = []

        # type: ignore
        for spec in self.config.two_step_optimizer_schedule['trunk_phase']:
            if isinstance(spec, dict):
                spec = OptimizerSpec(**spec)
            trunk_phase_optimizer = create_optimizer(
                spec=spec, params=trainable_params)
            trunk_phase_scheduler = create_scheduler(
                spec, trunk_phase_optimizer)
            self.trunk_train_schedule.append(
                (spec.epochs, trunk_phase_optimizer, trunk_phase_scheduler))

        # type: ignore
        for spec in self.config.two_step_optimizer_schedule['branch_phase']:
            if isinstance(spec, dict):
                spec = OptimizerSpec(**spec)
            branch_phase_optimizer = create_optimizer(
                spec=spec, params=trainable_params)
            branch_phase_scheduler = create_scheduler(
                spec, branch_phase_optimizer)
            self.branch_train_schedule.append(
                (spec.epochs, branch_phase_optimizer, branch_phase_scheduler))

    def _get_trainable_parameters(self, model: 'DeepONet'):
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

    def get_train_schedule(self) -> list[tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]]:
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

    def execute_phase_transition(self, model: 'DeepONet', full_branch_batch: torch.Tensor, full_trunk_batch: torch.Tensor):
        """Decomposes the trunk and updates the model for phase 2 training."""
        if not isinstance(self.config, TwoStepConfig):
            raise TypeError("TwoStepStrategy requires TwoStepConfig")

        if self._phase != 1:
            raise RuntimeError("Phase transition from invalid state")

        R, T_matrix = self._decompose_trunk(
            trunk=model.trunk, full_trunk_batch=full_trunk_batch)

        self.R = R
        self.A = model.branch
        self.all_branch_inputs = full_branch_batch

        new_trunk_config = deepcopy(self._original_trunk_cfg)
        new_trunk_config.component_type = "orthonormal_trunk"
        new_trunk_config.architecture = "pretrained"
        new_trunk_config.inner_config = deepcopy(
            self._original_trunk_cfg)  # Preserve original

        self.final_trunk_config = new_trunk_config
        self.final_branch_config = deepcopy(self._original_branch_cfg)

        self._original_branch_cfg.output_dim *= model.output_handler.num_channels
        if isinstance(model.output_handler, SplitOutputsHandler):
            self.final_trunk_config.output_dim *= model.output_handler.num_channels

        with torch.no_grad():
            self.A_full = self.A(self.all_branch_inputs)

        self.output_handler = model.output_handler

        if self._original_branch_cfg is None:
            raise RuntimeError("Missing original component configs.")

        new_trunk = OrthonormalTrunk(model.trunk, T_matrix)
        new_branch = BranchFactory.build(self._original_branch_cfg)

        model.trunk = new_trunk
        model.branch = new_branch

        model.trunk.requires_grad_(False)
        model.branch.requires_grad_(True)

        self._update_optimizer_parameters(model)

        self._phase = 2

    def _update_optimizer_parameters(self, model: 'DeepONet'):
        """Updates optimizer parameters to match current model parameters"""
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
        return False

    def should_transition_phase(self, current_phase: int, current_epoch: int) -> bool:
        if current_phase == 1:
            # Transition after completing trunk phase
            trunk_epochs = sum(epochs for epochs, _,
                               _ in self.trunk_train_schedule)
            return current_epoch >= trunk_epochs
        return False

    def compute_loss(self, model: DeepONet,
                     x_branch: torch.Tensor,
                     x_trunk: torch.Tensor,
                     y_true: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Computes the loss for the given model and data"""
        if self._phase == 1:
            y_pred = model(x_branch, x_trunk)
        elif self._phase == 2:
            y_true = self.compute_synthetic_targets(x_branch)
            y_pred = model.branch(x_branch)
        else:
            raise RuntimeError("Invalid training phase")
        loss = self.loss(y_pred, y_true)
        return y_pred, loss

    def calculate_metrics(self,
                          y_true: torch.Tensor,
                          y_pred: torch.Tensor,
                          loss: float,
                          train: bool,
                          branch_input: torch.Tensor,
                          label_map: list[str] | None = None) -> dict[str, float]:
        """Combines base and strategy-specific metrics"""
        if self._phase == 1:
            metrics = self.base_metrics(y_true, y_pred, loss, label_map)
        else:
            metrics = {'loss': loss}
        metrics.update(self.strategy_specific_metrics(
            y_true=y_true, y_pred=y_pred, branch_input=branch_input, label_map=label_map))
        return metrics

    def strategy_specific_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, branch_input: torch.Tensor, label_map: list[str] | None = None) -> dict[str, float]:
        if self._phase == 1:
            relative_error = self.error_metric(
                y_true - y_pred) / self.error_metric(y_true)
        elif self._phase == 2:
            y_true = self.compute_synthetic_targets(branch_input)
            if y_true.shape != y_pred.shape:
                raise ValueError(
                    "Synthetic targets and predictions must have the same shape.")
            relative_error = self.error_metric(
                y_true - y_pred) / self.error_metric(y_true)
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

    def compute_synthetic_targets(self, branch_inputs) -> torch.Tensor:
        """Computes synthetic targets using the decomposed trunk and branch"""
        if self.A is None or self.R is None:
            raise RuntimeError(
                "A and R matrices must be set before computing synthetic targets.")
        if self.output_handler is None:
            raise RuntimeError(
                "Output handler must be known by strategy before computing synthetic targets.")

        indices = self._find_branch_indices(branch_inputs)

        A_batch = self.A_full[indices]

        if isinstance(self.output_handler, SharedTrunkHandler):
            num_channels = self.output_handler.num_channels
            P = self.R.shape[0]  # Basis size
            A_reshaped = A_batch.view(-1, num_channels, P)  # (B, C, P)
            # (B, C, P) @ (P, P) -> (B, C, P)
            synthetic = torch.einsum('bcp,pq->bcq', A_reshaped, self.R.T)
            return synthetic.reshape(-1, num_channels * P)
        else:  # (B, C*P) @ (C*P, C*P) -> (B, C*P)
            return A_batch @ self.R.T

    def get_optimizer_scheduler(self):
        return self.config.optimizer_scheduler  # type: ignore

    def get_phases(self) -> list[str]:
        """Return phase names (e.g., ['phase1', 'phase2'])"""
        return ["trunk_phase", "branch_phase"]

    def apply_gradient_constraints(self, model: DeepONet):
        """Optional gradient clipping/normalization"""
        pass

    def _find_branch_indices(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        """Finds indices of batch inputs in full branch set"""
        # Compute distances (memory efficient for large datasets)
        dists = torch.cdist(batch_inputs, self.all_branch_inputs)
        min_dists = torch.min(dists, dim=1).values
        return torch.where(min_dists < 1e-6, torch.argmin(dists, dim=1), -1)

    def _decompose_trunk(self, trunk: torch.nn.Module, full_trunk_batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """SVD decomposition of final trunk layer"""
        with torch.no_grad():
            phi_matrix = trunk(full_trunk_batch)
        if self.config.decomposition_type.lower() == "qr":  # type: ignore
            Q, R = torch.linalg.qr(phi_matrix)
        elif self.config.decomposition_type.lower() == "svd":  # type: ignore
            Q, S, V = torch.svd(phi_matrix)
            R = torch.diag(S) @ V

        else:
            raise NotImplementedError(
                # type: ignore
                f"Decomposition type '{self.config.decomposition_type}' is not implemented."
            )
        T = torch.linalg.pinv(R)
        return R, T
