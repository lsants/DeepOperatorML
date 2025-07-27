from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal, Optional, Callable
from src.modules.model.components.registry import ComponentRegistry
from src.modules.data_processing.config import TransformConfig
from src.modules.model.nn.activation_functions.activation_fns import ACTIVATION_MAP


@dataclass
class BranchConfig:
    # Fundamental component type
    architecture: Optional[Literal["resnet",
                                   "mlp",
                                   "chebyshev_kan",
                                   "trainable_matrix"]]
    component_type: Literal["neural_branch", "matrix_branch"] = "neural_branch"
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    # Architecture-specific parameters (for neural)
    hidden_layers: Optional[list[int]] = None
    dropout_rates: Optional[list[float]] = None
    batch_normalization: Optional[list[bool]] = None
    layer_normalization: Optional[list[bool]] = None
    activation: Optional[Callable | str] = None
    degree: Optional[int] = None

    @classmethod
    def setup_for_training(cls, data_cfg: dict, train_cfg: dict) -> "BranchConfig":
        branch_config = BranchConfig(**train_cfg["branch"])
        if branch_config.activation is not None:
            branch_config.activation = ACTIVATION_MAP[branch_config.activation.lower(
            )]
        branch_config.input_dim = data_cfg["shapes"][data_cfg["features"][0]][1]
        branch_config.output_dim = train_cfg["embedding_dimension"]
        return branch_config

    @classmethod
    def setup_for_inference(cls, model_cfg_dict: dict, transform_cfg: TransformConfig) -> "BranchConfig":
        branch_config = BranchConfig(**model_cfg_dict["branch"])
        if branch_config.activation is not None:
            mask = re.sub(r'[^a-zA-Z0-9]', '', branch_config.activation.lower(
            ))
            branch_config.activation = ACTIVATION_MAP[mask]
        if transform_cfg.branch.feature_expansion.size is None:
            transform_cfg.branch.feature_expansion.size = 0
        branch_config.input_dim = transform_cfg.branch.original_dim * (1 +
                                                                       transform_cfg.branch.feature_expansion.size)
        return branch_config


class BranchConfigValidator:
    @staticmethod
    def validate(config: BranchConfig):
        try:
            component_class, required_params = ComponentRegistry.get(
                component_type=config.component_type,
                architecture=config.architecture
            )

            required_params = [p for p in required_params if p != "self"]

            missing = [p for p in required_params if not hasattr(config, p)]
            if missing:
                raise ValueError(
                    f"Missing required parameters for {config.architecture}: {missing}"
                )

        except ValueError as e:
            raise ValueError(f"Invalid branch configuration: {e}") from e

        # Architecture-specific validation
        if 'kan' in str(config.architecture) and config.degree < 1:  # type: ignore
            raise ValueError("KAN requires degree >= 1")
