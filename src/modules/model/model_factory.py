from __future__ import annotations
import os
import yaml
import torch
import logging
from typing import Any
from .deeponet import DeepONet
from .config import ModelConfig
from .components.component_factory import branch_factory, trunk_factory

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def create(config: ModelConfig) -> DeepONet:
        # 1. Create base components
        branch = branch_factory(config.branch)
        trunk = trunk_factory(config.trunk)
        
        # 2. Apply output handling
        config.output.configure(branch, trunk)
        
        # 3. Apply training strategy
        config.strategy.configure_components(branch, trunk)
        
        # 4. Assemble
        return DeepONet(
            branch=branch,
            trunk=trunk,
            strategy=config.strategy,
            output=config.output
        )