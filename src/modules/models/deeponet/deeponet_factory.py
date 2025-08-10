from __future__ import annotations
import torch
import logging
import dataclasses
from dataclasses import fields
from src.modules.models.deeponet.deeponet import DeepONet
from src.modules.models.deeponet.components.rescaling.rescaler import Rescaler
from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig
from src.modules.models.deeponet.components.output_handler.registry import OutputRegistry
from src.modules.models.deeponet.components.bias.config import BiasConfigValidator
from src.modules.models.deeponet.components.component_factory import BiasFactory
from src.modules.models.deeponet.components.component_factory import TrunkFactory
from src.modules.models.deeponet.components.component_factory import BranchFactory
from src.modules.models.deeponet.training_strategies.base import TrainingStrategy
from src.modules.models.deeponet.components.trunk.config import TrunkConfigValidator
from src.modules.models.deeponet.components.branch.config import BranchConfigValidator
from src.modules.models.deeponet.training_strategies.pod_strategy import PODStrategy
from src.modules.models.deeponet.training_strategies.vanilla_strategy import VanillaStrategy
from src.modules.models.deeponet.training_strategies.two_step_strategy import TwoStepStrategy
from src.modules.models.deeponet.training_strategies.config import VanillaConfig, PODConfig, TwoStepConfig

logger = logging.getLogger(__name__)

class DeepONetFactory:
    """
    A factory class for creating and configuring DeepONet models.

    This class provides static methods to build a DeepONet model either for
    training with an associated training strategy or for inference from a
    pre-trained state. It handles the instantiation of model components,
    configuration validation, and state loading.
    """
    @classmethod
    def create_for_training(cls, config: DeepONetConfig) -> tuple[DeepONet, TrainingStrategy]:
        """
        Creates a DeepONet model and a corresponding training strategy based on a configuration.

        This method validates the configuration, builds the model components (branch, trunk,
        and bias networks), and instantiates the appropriate training strategy.

        Args:
            config (DeepONetConfig): A configuration object containing all necessary
                                     parameters for the DeepONet and the training strategy.

        Returns:
            tuple[DeepONet, TrainingStrategy]: A tuple containing the initialized DeepONet
                                               model and its associated training strategy.
        """
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
    def create_for_inference(cls, saved_config: DeepONetConfig, state_dict: dict) -> DeepONet:
        """
        Builds a DeepONet model for inference from a saved configuration and state dictionary.

        This method constructs the model from a post-training configuration, loads the
        weights from the provided state dictionary, and sets the model to evaluation mode.
        It does not involve a training strategy.

        Args:
            saved_config (DeepONetConfig): The configuration object saved after training.
            state_dict (dict): The state dictionary containing the trained model weights.

        Returns:
            DeepONet: The fully constructed DeepONet model ready for inference.
        """
        cls._validate_inference_config(saved_config)

        output_handler = OutputRegistry.create(saved_config.output)

        # output_handler.adjust_dimensions(
        #     saved_config)
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
    def _validate_inference_config(cls, config: DeepONetConfig):
        """
        Ensures the inference configuration is valid and contains a post-training state.

        This method performs specific checks based on the training strategy used to ensure
        the model can be correctly initialized for inference.

        Args:
            config (DeepONetConfig): The configuration to be validated.

        Raises:
            ValueError: If the configuration is not valid for inference (e.g., missing
                        basis information for POD trunk).
        """
        if config.strategy.name != 'pod' and not config.output.dims_adjust:
            raise ValueError(
                "Inference config must retain output handler basis adjustment"
            )
        if config.strategy.name == "two_step":
            if config.trunk.architecture != "pretrained" or config.trunk.component_type != 'orthonormal_trunk':
                raise ValueError(
                    "Two-step inference requires pretrained decomposed trunk architecture"
                )
            if config.output.handler_type != "two_step_final" or config.trunk.component_type != 'orthonormal_trunk':
                raise ValueError(
                    "Two-step inference requires pretrained decomposed trunk architecture"
                )
        if config.strategy.name == "pod":
            if config.trunk.architecture != "precomputed" or config.trunk.component_type != 'pod_trunk':
                raise ValueError(
                    f"POD inference requires precomputed POD trunk architecture, has {config.trunk.architecture}"
                )
            if config.trunk.pod_basis_shape is None:
                raise ValueError(
                    "Shape of precomputed POD basis must be known in order to initialize model."
                )

            config.trunk.pod_basis = torch.rand(config.trunk.pod_basis_shape)
        else:
            TrunkConfigValidator.validate(config.trunk)
            BranchConfigValidator.validate(config.branch)

    @classmethod
    def _create_strategy(cls, config: dict) -> TrainingStrategy:
        """
        Initializes and returns a specific training strategy based on the configuration.

        Args:
            config (dict): A dictionary containing the `strategy` configuration details.

        Returns:
            TrainingStrategy: An instance of the requested training strategy.
        """
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
