from __future__ import annotations
import dataclasses
import logging
from copy import deepcopy
from dataclasses import fields, is_dataclass
from .config import ModelConfig
from .deeponet import DeepONet
from typing import TYPE_CHECKING, Any
from .components.component_factory import TrunkFactory
from .components.component_factory import BranchFactory
from ...exceptions import ConfigValidationError
from .components.rescaling.rescaler import Rescaler
from .training_strategies.base import TrainingStrategy
from .components.trunk.config import TrunkConfigValidator
from .components.branch.config import BranchConfigValidator
from .components.output_handler.registry import OutputRegistry
from .training_strategies.vanilla_training import VanillaStrategy
from .training_strategies.two_step_training import TwoStepStrategy
from .training_strategies.pod_strategy import PODStrategy
from .training_strategies.config import StrategyConfig, VanillaConfig, PODConfig, TwoStepConfig
# if TYPE_CHECKING:

logger = logging.getLogger(__name__)

class ModelFactory:
    @classmethod
    def create_for_training(cls, config: ModelConfig) -> tuple[DeepONet, TrainingStrategy]:
        # 1. Clone config to prevent mutation of original
        model_config = deepcopy(config)
        
        # 2. Instantiate strategy
        strategy = cls._create_strategy(config=dataclasses.asdict(model_config))
        strategy.prepare_components(model_config)

        # 3. Output handler adjusts dimensions (e.g., trunk.output_dim *= channels)
        output_handler = OutputRegistry.create(model_config.output)
        output_handler.adjust_dimensions(model_config)  # (2) Modifies numeric fields
        
        # 4. Validate config post-adjustments
        BranchConfigValidator.validate(model_config.branch)
        TrunkConfigValidator.validate(model_config.trunk)
        
        # 5. Build components with finalized dimensions
        branch = BranchFactory.build(model_config.branch)
        trunk = TrunkFactory.build(model_config.trunk)
        
        # 6. Assemble with strategy-aware components
        return (DeepONet(
                        branch=branch,
                        trunk=trunk,
                        output_handler=output_handler,
                        rescaler=Rescaler(model_config.rescaling)
                    ),
                strategy
                )

    @classmethod
    def create_for_inference(cls, saved_config: ModelConfig) -> DeepONet:
        """Builds model from frozen post-training config without strategy involvement"""
        # 1. Validate config completeness for inference
        cls._validate_inference_config(saved_config)
        
        # 2. Build components directly from frozen config
        branch = BranchFactory.build(saved_config.branch)
        trunk = TrunkFactory.build(saved_config.trunk)
        
        # 3. Initialize non-trainable components
        output_handler = OutputRegistry.create(saved_config.output)
        rescaler = Rescaler(saved_config.rescaling)
        
        # 4. Assemble inference-ready model
        return DeepONet(
            branch=branch,
            trunk=trunk,
            output_handler=output_handler,
            rescaler=rescaler
        )
    
    @classmethod
    def _validate_inference_config(cls, config: ModelConfig):
        """Ensures config contains post-training state"""
        # Check for decomposed trunk in two-step models
        if config.strategy == "two_step":
            if config.trunk.architecture != "decomposed":
                raise ConfigValidationError(
                    "Two-step inference requires decomposed trunk architecture"
                )
            if "basis" not in config.trunk.__dict__:
                raise ConfigValidationError(
                    "Decomposed trunk missing basis tensor in inference config"
                )

        # Validate all components
        BranchConfigValidator.validate(config.branch)
        TrunkConfigValidator.validate(config.trunk)
        
        # Output handler must match training configuration
        if not config.output.basis_adjust:
            raise ConfigValidationError(
                "Inference config must retain output handler basis adjustment"
            )
    @classmethod
    def _create_strategy(cls, config: dict) -> TrainingStrategy:
        strategy_map = {
            "vanilla": (VanillaStrategy, VanillaConfig),
            "pod": (PODStrategy, PODConfig),
            "two_step": (TwoStepStrategy, TwoStepConfig)
        }
        
        strategy_name = config['strategy']['name']
        strategy_class, config_class = strategy_map[strategy_name]
        
        # Get valid fields for the target config class
        valid_fields = {f.name for f in fields(config_class)}
        
        # Filter input config to only include valid fields
        filtered_config = {k: v for k, v in config["strategy"].items() if k in valid_fields}


        # Instantiate strategy-specific config
        validated_config = config_class(**filtered_config)

        return strategy_class(validated_config)