import torch
from .training_strategy_base import TrainingStrategy

class StandardTrainingStrategy(TrainingStrategy):
    def __init__(self):
        super().__init__()
        self.phases = ['default']

    def get_epochs(self, params):
        return [params['EPOCHS']]

    def prepare_training(self, model, **kwargs):
        if hasattr(model, 'trunk_networks'):
            for trunk in model.trunk_networks:
                for param in trunk.parameters():
                    param.requires_grad = True
        if hasattr(model, 'branch_networks'):
            for branch in model.branch_networks:
                for param in branch.parameters():
                    param.requires_grad = True

 