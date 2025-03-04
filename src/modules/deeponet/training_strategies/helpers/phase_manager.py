import torch
import logging
from typing import TYPE_CHECKING, Any, Dict, Tuple, Optional
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

logger = logging.getLogger(__name__)

class PhaseManager:
    def __init__(self) -> None:
        # Only two phases are used in training.
        self.current_phase = 'trunk'

    def prepare_phase(self, model: 'DeepONet') -> None:
        """Adjust model parameters based on the current phase."""
        if self.current_phase == 'trunk':
            self._freeze_branch(model)
            self._unfreeze_trunk(model)
        elif self.current_phase == 'branch':
            self._freeze_trunk(model)
            self._unfreeze_branch(model)

    def update_phase(self, new_phase: str) -> None:
        """Updates the phase; allowed values are 'trunk' and 'branch'."""
        if new_phase not in ['trunk', 'branch']:
            raise ValueError("Invalid phase. Allowed phases: 'trunk' and 'branch'.")
        self.current_phase = new_phase
        logger.info(f"Training phase updated to: {self.current_phase}")

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