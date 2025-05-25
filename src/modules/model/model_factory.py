from __future__ import annotations
import logging
from copy import deepcopy
from .config import ModelConfig
from .deeponet import DeepONet
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
from .training_strategies.base import StrategyConfig

logger = logging.getLogger(__name__)

class ModelFactory:
    @classmethod
    def create_for_training(cls, config: ModelConfig) -> DeepONet:
        # 1. Clone config to prevent mutation of original
        train_config = deepcopy(config)
        
        # 2. Instantiate strategy
        strategy = cls._create_strategy(config=train_config.strategy)
        strategy.prepare_components(train_config)

        # 3. Output handler adjusts dimensions (e.g., trunk.output_dim *= channels)
        output_handler = OutputRegistry.create(train_config.output)
        output_handler.adjust_dimensions(train_config)  # (2) Modifies numeric fields
        
        # 4. Validate config post-adjustments
        BranchConfigValidator.validate(train_config.branch)
        TrunkConfigValidator.validate(train_config.trunk)
        
        # 5. Build components with finalized dimensions
        branch = BranchFactory.build(train_config.branch)
        trunk = TrunkFactory.build(train_config.trunk)
        
        # 6. Assemble with strategy-aware components
        return DeepONet(
            branch=branch,
            trunk=trunk,
            output_handler=output_handler,
            rescaler=Rescaler(train_config.rescaling)
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
    def _create_strategy(cls, config: StrategyConfig) -> TrainingStrategy:
        strategy_map = {
            "vanilla": (VanillaStrategy, StrategyConfig),
            "two_step": (TwoStepStrategy, StrategyConfig),
            "pod": (PODStrategy, StrategyConfig)
        }
        strategy_class, config_class = strategy_map[config.name]
        validated_config = config_class(**config.__dict__)
        return strategy_class(validated_config)