import torch
from copy import deepcopy
from .base import TrainingStrategy
from typing import TYPE_CHECKING
from .config import TwoStepConfig
from ..optimization.optimizers.optimizer_factory import create_optimizer, create_scheduler
if TYPE_CHECKING:
    from ..deeponet import DeepONet
    from ..config import ModelConfig
    from ..components.trunk.decomposed_trunk import DecomposedTrunk
    from ..components.component_factory import BranchFactory, TrunkFactory


class TwoStepStrategy(TrainingStrategy):
    def __init__(self, config: TwoStepConfig):
        super().__init__(config)
        self._phase = 1
        self._original_branch_cfg = None
        self._original_trunk_cfg = None

    def prepare_components(self, model_config: 'ModelConfig'):
        self._original_branch_cfg = deepcopy(model_config.branch)
        self._original_trunk_cfg = deepcopy(model_config.trunk)

        # Phase 1 components
        model_config.branch.component_type = "matrix"
        model_config.trunk.component_type = "trunk_neural"

    def setup_training(self, model: 'DeepONet'):
        # Phase 1: Both components trainable
        model.trunk.requires_grad_(True)
        model.branch.requires_grad_(True)
        trainable_params = self._get_trainable_parameters(model)
        if not trainable_params:
            raise ValueError("No trainable parameters found in the model.")
        self.trunk_train_schedule = []
        self.branch_train_schedule = []
        for spec in self.config.two_step_optimizer_scheduler['trunk_phase']:
            trunk_phase_optimizer = create_optimizer(spec, trainable_params)
            trunk_phase_scheduler = create_scheduler(spec, trunk_phase_optimizer)
            self.trunk_train_schedule.append((trunk_phase_optimizer, trunk_phase_scheduler, spec.epochs))
        for spec in self.config.two_step_optimizer_scheduler['branch_phase']:
            branch_phase_optimizer = create_optimizer(spec, trainable_params)
            branch_phase_scheduler = create_scheduler(spec, branch_phase_optimizer)
            self.branch_train_schedule.append((branch_phase_optimizer, branch_phase_scheduler, spec.epochs))


    def _get_trainable_parameters(self, model: 'DeepONet'):
        trainable_params = []
        for name, param in model.trunk.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        for name, param in model.branch.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def get_train_schedule(self) -> list[tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]]:
        if self._phase == 1:
            if not hasattr(self, 'trunk_train_schedule'):
                raise ValueError("Training schedule for trunk not set up. Call setup_training first.")
            return self.trunk_train_schedule
        elif self._phase == 2:
            if not hasattr(self, 'branch_train_schedule'):
                raise ValueError("Training schedule for branch not set up. Call setup_training first.")
            return self.branch_train_schedule
        else:
            raise RuntimeError("Invalid training phase")

    def check_phase_transition(self, epoch: int) -> bool:
        if not isinstance(self.config, TwoStepConfig):
            raise TypeError("TwoStepStrategy requires TwoStepConfig")
        if self._phase == 1:
            return epoch == self.config.trunk_epochs
        else:
            return epoch == self.config.trunk_epochs + self.config.branch_epochs + 1


    def execute_phase_transition(self, model: 'DeepONet', full_trunk_batch: torch.Tensor):
        """Decomposes the trunk and updates the model for phase 2 training."""
        if not isinstance(self.config, TwoStepConfig):
            raise TypeError("TwoStepStrategy requires TwoStepConfig")
        # 1. Validate phase state
        if self._phase != 1:
            raise RuntimeError("Phase transition from invalid state")

        # 2. Decompose trunk
        basis = self._decompose_trunk(trunk=model.trunk, full_trunk_batch=full_trunk_batch)

        # 3. Update config for phase 2
        if self._original_trunk_cfg is not None and self._original_branch_cfg is not None:
            new_trunk_config = deepcopy(self._original_trunk_cfg)
            new_trunk_config.component_type = "decomposed"
            new_trunk_config.basis = basis
        else:
            raise RuntimeError("Missing original component configs.")

        # 4. Rebuild components
        new_trunk = TrunkFactory.build(new_trunk_config)
        new_branch = BranchFactory.build(self._original_branch_cfg)

        # 5. Replace components
        model.trunk = new_trunk
        model.branch = new_branch

        # 6. Freeze trunk
        model.trunk.requires_grad_(False)

        self._phase = 2

    def validation_enabled(self) -> bool:
        return False

    def _decompose_trunk(self, trunk: torch.nn.Module, full_trunk_batch: torch.Tensor) -> torch.Tensor:
        """SVD decomposition of final trunk layer"""
        phi_matrix = self._get_phi_matrix(trunk=trunk, full_trunk_batch=full_trunk_batch)
        if self.config.decomposition_type.lower() == "qr": # type: ignore
            Q, R = torch.linalg.qr(torch.tensor(phi_matrix))
        elif self.config.decomposition_type.lower() == "svd": # type: ignore
            Q, S, V = torch.svd(torch.tensor(phi_matrix))
            R = torch.diag(S) @ V
        else:
            raise NotImplementedError(
                f"Decomposition type '{self.config.decomposition_type}' is not implemented." # type: ignore
            )
        T = torch.linalg.pinv(R)
        return Q @ R @ T

    def _get_phi_matrix(self, trunk: torch.nn.Module, full_trunk_batch: torch.Tensor) -> torch.Tensor:
        """Extracts the phi matrix from the trunk's final layer"""
        return trunk(full_trunk_batch)