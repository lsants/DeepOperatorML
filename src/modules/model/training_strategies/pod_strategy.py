# File: src/modules/deeponet/training_strategies/pod_training_strategy.py
from __future__ import annotations
import logging
import torch

from src.modules.pipe.pipeline_config import ComponentConfig, DataConfig, TrainingConfig
from ..deeponet import DeepONet
from .helpers import PODBasisHelper
from collections.abc import Callable, Iterable
from typing import Any
from ..components.pod_trunk import PODTrunk
from .base import TrainingStrategy, PhaseConfig
from 

logger = logging.getLogger(name=__name__)

class PODStrategy(TrainingStrategy):
    def __init__(self, train_cfg: TrainingConfig, data_cfg: DataConfig):
        super().__init__(train_cfg=train_cfg, data_cfg=data_cfg)
        self.params = train_cfg.strategy_params.pod

    def initialize_components(self):
        self.branch = ComponentRegistry.create(
            self.train_cfg.branch.architecture,
            ComponentConfig(**self.train_cfg.branch.__dict__)
        )
        self.trunk = ComponentRegistry.create(
            'pod',  # Force POD trunk
            ComponentConfig(architecture='pod',
                            params={'input_dim': len(self.data_cfg.features[1])}
                            )
        )
    
    def get_phases(self):
        return [
            PhaseConfig(
                name='pod', 
                epochs=self.train_cfg.strategy_params.pod.epochs, 
                trainable_components=['branch']
            )
        ]
    
    def preprocess_data(self, data):
        self.trunk.get_basis(data, self.train_cfg.strategy_params.pod.var_share)
    
    def configure_output_handling(self):
        # POD requires fixed output dimensions
        self.output_handler = FixedBasisHandler(self.trunk.basis)

