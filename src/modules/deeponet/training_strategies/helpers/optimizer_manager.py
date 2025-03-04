import torch
import logging
from ...deeponet import DeepONet
from typing import Any, Dict, Tuple, Optional
from training_strategy_base import TrainingStrategy


class OptimizerSchedulerHelper:
    def get_optimizers(self, model: DeepONet, params: Dict[str, Any], A: Optional[torch.nn.Parameter]) -> Dict[str, torch.optim.Optimizer]:
        """Creates separate optimizers for trunk and branch networks."""
        trunk_params = list(model.trunk_network.parameters())
        if A is not None:
            trunk_params.append(A)
        optimizers = {
            'trunk': torch.optim.Adam(
                trunk_params,
                lr=params.get('TRUNK_LEARNING_RATE'),
                weight_decay=params.get('L2_REGULARIZATION', 0)
            ),
            'branch': torch.optim.Adam(
                model.branch_network.parameters(),
                lr=params.get('BRANCH_LEARNING_RATE'),
                weight_decay=params.get('L2_REGULARIZATION', 0)
            )
        }
        return optimizers

    def get_schedulers(self, optimizers: Dict[str, torch.optim.Optimizer], params: Dict[str, Any]) -> Dict[str, Any]:
        """Creates learning rate schedulers for trunk and branch optimizers if enabled."""
        schedulers = {}
        if params.get("LR_SCHEDULING", False):
            schedulers['trunk'] = torch.optim.lr_scheduler.StepLR(
                optimizers['trunk'],
                step_size=params.get('TRUNK_SCHEDULER_STEP_SIZE'),
                gamma=params.get('TRUNK_SCHEDULER_GAMMA')
            )
            schedulers['branch'] = torch.optim.lr_scheduler.StepLR(
                optimizers['branch'],
                step_size=params.get('BRANCH_SCHEDULER_STEP_SIZE'),
                gamma=params.get('BRANCH_SCHEDULER_GAMMA')
            )
        return schedulers