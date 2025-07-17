from __future__ import annotations
import torch
import logging
import dataclasses
from copy import deepcopy
from dataclasses import fields

from src.modules.model.components.bias.config import BiasConfigValidator
from .config import ModelConfig
from .deeponet import DeepONet
from .components.component_factory import BiasFactory
from .components.component_factory import TrunkFactory
from .components.component_factory import BranchFactory
from .components.rescaling.rescaler import Rescaler
from .training_strategies.base import TrainingStrategy
from .components.trunk.config import TrunkConfigValidator
from .components.branch.config import BranchConfigValidator
from .components.output_handler.registry import OutputRegistry
from .training_strategies.vanilla_strategy import VanillaStrategy
from .training_strategies.two_step_strategy import TwoStepStrategy
from .training_strategies.config import VanillaConfig, PODConfig, TwoStepConfig
from .training_strategies.pod_strategy import PODStrategy

logger = logging.getLogger(__name__)


class ModelFactory:
    @classmethod
    def create_for_training(cls, config: ModelConfig) -> tuple[DeepONet, TrainingStrategy]:

        model_config = deepcopy(config)

        strategy = cls._create_strategy(
            config=dataclasses.asdict(model_config))
        strategy.prepare_components(model_config)

        output_handler = OutputRegistry.create(model_config.output)
        output_handler.adjust_dimensions(
            model_config)

        rescaler = Rescaler(model_config.rescaling)

        BiasConfigValidator.validate(model_config.bias)
        BranchConfigValidator.validate(model_config.branch)
        TrunkConfigValidator.validate(model_config.trunk)

        branch = BranchFactory.build(model_config.branch)
        trunk = TrunkFactory.build(model_config.trunk)
        bias = BiasFactory.build(model_config.bias)

        return (DeepONet(
            branch=branch,
            trunk=trunk,
            bias=bias,
            output_handler=output_handler,
            rescaler=rescaler
        ),
            strategy
        )

    @classmethod
    def create_for_inference(cls, saved_config: ModelConfig, state_dict: dict) -> DeepONet:
        """Builds model from frozen post-training config without strategy involvement"""
        cls._validate_inference_config(saved_config)

        output_handler = OutputRegistry.create(saved_config.output)
        output_handler.adjust_dimensions(
            saved_config)

        rescaler = Rescaler(saved_config.rescaling)

        trunk = TrunkFactory.build(saved_config.trunk)
        branch = BranchFactory.build(saved_config.branch)
        bias = BiasFactory.build(saved_config.bias)

        model = DeepONet(
            branch=branch,
            trunk=trunk,
            bias=bias,
            output_handler=output_handler,
            rescaler=rescaler
        )

        model.load_state_dict(state_dict)
        model.eval()

        return model

    @classmethod
    def _validate_inference_config(cls, config: ModelConfig):
        """Ensures config contains post-training state"""
        BranchConfigValidator.validate(config.branch)
        if config.strategy.name != 'pod' and not config.output.basis_adjust:
            raise ValueError(
                "Inference config must retain output handler basis adjustment"
            )
        if config.strategy.name == "two_step":
            if config.trunk.architecture != "pretrained" or config.trunk.component_type != 'orthonormal_trunk':
                raise ValueError(
                    "Two-step inference requires pretrained decomposed trunk architecture"
                )
        if config.strategy.name == "pod":
            if config.trunk.architecture != "precomputed" or config.trunk.component_type != 'pod_trunk':
                raise ValueError(
                    "POD inference requires precomputed POD trunk architecture"
                )
            if config.trunk.pod_basis_shape is None:
                raise ValueError(
                    "Shape of precomputed POD basis must be known in order to initialize model."
                )
            config.trunk.pod_basis = torch.rand(config.trunk.pod_basis_shape)
            if config.output.basis_adjust:
                config.output.basis_adjust = False
        else:
            TrunkConfigValidator.validate(config.trunk)

    @classmethod
    def _create_strategy(cls, config: dict) -> TrainingStrategy:
        strategy_map = {
            "vanilla": (VanillaStrategy, VanillaConfig),
            "pod": (PODStrategy, PODConfig),
            "two_step": (TwoStepStrategy, TwoStepConfig)
        }

        strategy_name = config['strategy']['name']
        strategy_class, config_class = strategy_map[strategy_name]

        valid_fields = {f.name for f in fields(config_class)}

        filtered_config = {
            k: v for k, v in config["strategy"].items() if k in valid_fields}

        validated_config = config_class(**filtered_config)

        return strategy_class(validated_config)  # type: ignore
