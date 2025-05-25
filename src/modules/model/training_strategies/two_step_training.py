import torch
from copy import deepcopy
from typing import List, Dict
from dataclasses import dataclass
from .base import TrainingStrategy, StrategyConfig
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..deeponet import DeepONet
    from ..config import ModelConfig
    from ..components.trunk.decomposed_trunk import DecomposedTrunk
    from ..components.component_factory import BranchFactory, TrunkFactory


class TwoStepStrategy(TrainingStrategy):
    def __init__(self, config: StrategyConfig):
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

    def check_phase_transition(self, model: 'DeepONet', epoch: int) -> bool:
        if not isinstance(self.config, StrategyConfig):
            raise TypeError("TwoStepStrategy requires TwoStepConfig")
        return epoch == self.config.two_step_trunk_epochs

    def execute_phase_transition(self, model: 'DeepONet'):
        # 1. Validate phase state
        if self._phase != 1:
            raise RuntimeError("Phase transition from invalid state")

        # 2. Decompose trunk
        basis = self._decompose_trunk(model.trunk)

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

    def _decompose_trunk(self, trunk: torch.nn.Module) -> torch.Tensor:
        """SVD decomposition of final trunk layer"""
        weights = trunk.net[-1].weight.data
        U, _, _ = torch.svd(weights.T)
        return U[:, :self._original_trunk_cfg.output_dim]
