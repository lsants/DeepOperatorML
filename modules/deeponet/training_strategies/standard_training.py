import torch
from .training_strategy_base import TrainingStrategy
from ..loss_functions.loss_complex import loss_complex

class StandardTrainingStrategy(TrainingStrategy):
    def __init__(self):
        super().__init__()
        self.phases = ['final']

    def get_epochs(self, params):
        return [params['EPOCHS']]

    def update_training_phase(self, phase):
        if phase != 'final':
            raise ValueError(f"Invalid phase for StandardTrainingStrategy: {phase}")
        self.current_phase = phase

    def prepare_training(self, model, **kwargs):
        for trunk in model.trunk_networks:
            for param in trunk.parameters():
                param.requires_grad = True
        for branch in model.branch_networks:
            for param in branch.parameters():
                param.requires_grad = True

 