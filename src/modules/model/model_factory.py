from __future__ import annotations
import torch
import logging
import dataclasses
from dataclasses import fields

from src.modules.model.components.bias.config import BiasConfigValidator
from src.modules.model.config import ModelConfig
from src.modules.model.deeponet import DeepONet
from src.modules.model.components.component_factory import BiasFactory
from src.modules.model.components.component_factory import TrunkFactory
from src.modules.model.components.component_factory import BranchFactory
from src.modules.model.components.rescaling.rescaler import Rescaler
from src.modules.model.training_strategies.base import TrainingStrategy
from src.modules.model.components.trunk.config import TrunkConfigValidator
from src.modules.model.components.branch.config import BranchConfigValidator
from src.modules.model.components.output_handler.registry import OutputRegistry
from src.modules.model.training_strategies.vanilla_strategy import VanillaStrategy
from src.modules.model.training_strategies.two_step_strategy import TwoStepStrategy
from src.modules.model.training_strategies.config import VanillaConfig, PODConfig, TwoStepConfig
from src.modules.model.training_strategies.pod_strategy import PODStrategy

logger = logging.getLogger(__name__)

class ModelFactory:
    @classmethod
    def create_for_training(cls, config: ModelConfig) -> tuple[DeepONet, TrainingStrategy]:
        strategy = cls._create_strategy(
            config=dataclasses.asdict(config))
        strategy.prepare_components(config)
        output_handler = OutputRegistry.create(config.output)
        output_handler.adjust_dimensions(config)

        rescaler = Rescaler(config.rescaling)

        BiasConfigValidator.validate(config.bias)
        BranchConfigValidator.validate(config.branch)
        TrunkConfigValidator.validate(config.trunk)

        branch = BranchFactory.build(config.branch)
        trunk = TrunkFactory.build(config.trunk)
        bias = BiasFactory.build(config.bias)

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
        if config.strategy.name != 'pod' and not config.output.dims_adjust:
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
            if config.rescaling.embedding_dimension * config.output.num_channels == config.branch.output_dim:
                if config.output.dims_adjust:
                    config.output.dims_adjust = False

            config.trunk.pod_basis = torch.rand(config.trunk.pod_basis_shape)
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
