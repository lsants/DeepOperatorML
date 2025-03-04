import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.deeponet.deeponet import DeepONet

from .training_strategy_base import TrainingStrategy

class StandardTrainingStrategy(TrainingStrategy):
    def __init__(self, loss_fn):
        super().__init__(loss_fn)
        self.phases = ['default']

    def get_epochs(self, params):
        return [params['EPOCHS']]

    def prepare_training(self, model: 'DeepONet', **kwargs):
        if hasattr(model, 'trunk_network'):
            for param in model.trunk_network.parameters():
                param.requires_grad = True
        if hasattr(model, 'branch_network'):
            for param in model.branch_network.parameters():
                param.requires_grad = True

 