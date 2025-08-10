from __future__ import annotations
import re
import torch
from dataclasses import dataclass
from typing import Literal, Optional, Callable
from src.modules.models.deeponet.components.registry import ComponentRegistry
from src.modules.models.deeponet.dataset.transform_config import TransformConfig
from src.modules.models.tools.activation_functions.activation_fns import ACTIVATION_MAP

@dataclass
class BranchConfig:
    # Fundamental component type
    architecture: Optional[Literal["resnet",
                                   "mlp",
                                   "chebyshev_kan",
                                   "trainable_matrix",
                                   "pretrained"]]
    component_type: Literal["neural_branch", "matrix_branch", "orthonormal_branch"] = "neural_branch"
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    # Architecture-specific parameters (for neural)
    hidden_layers: Optional[list[int]] = None
    dropout_rates: Optional[list[float]] = None
    batch_normalization: Optional[list[bool]] = None
    layer_normalization: Optional[list[bool]] = None
    activation: Optional[Callable | str] = None
    degree: Optional[int] = None
    inner_config: Optional[BranchConfig] = None
    R_matrix: Optional[torch.Tensor] = None
    num_channels: Optional[int] = None
    is_shared_branch: Optional[bool] = None

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
        if hasattr(transform_cfg.branch, 'feature_expansion'):
            if transform_cfg.branch.feature_expansion is not None:
                if transform_cfg.branch.feature_expansion.size is None:
                    transform_cfg.branch.feature_expansion.size = 0
                else:
                    if transform_cfg.branch.original_dim is not None:
                        branch_config.input_dim = transform_cfg.branch.original_dim * (1 + transform_cfg.branch.feature_expansion.size)
        if model_cfg_dict['strategy']['name'] == 'two_step':
            branch_config.inner_config = BranchConfig(**model_cfg_dict["branch"]["inner_config"])
            branch_config.inner_config.num_channels = model_cfg_dict['output']['num_channels']
            mask = re.sub(r'[^a-zA-Z0-9]', '', branch_config.inner_config.activation.lower(
            ))
            branch_config.inner_config.activation = ACTIVATION_MAP[mask]
            if model_cfg_dict['output']['handler_type'] == 'shared_branch':
                branch_config.inner_config.is_shared_branch = True
            else:
                branch_config.inner_config.is_shared_branch = False
        return branch_config


class BranchConfigValidator:
    @staticmethod
    def validate(config: BranchConfig):
        if config.component_type == "orthonormal_branch":
            BranchConfigValidator._validate_orthonormal(config)
            return
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
    
    @staticmethod
    def _validate_orthonormal(config: BranchConfig):
        """Special validation for orthonormal branch configuration"""
        errors = []

        if not hasattr(config, "inner_config"):
            errors.append("Missing inner_config for orthonormal branch")
        if not hasattr(config, "R_matrix"):
            errors.append("Missing coefficient matrix for orthonormal branch")

        if hasattr(config, "R_matrix"):
            R_matrix = config.R_matrix
            if not isinstance(R_matrix, (torch.Tensor)):
                errors.append("Basis matrix must be Tensor")
            elif R_matrix is None:
                errors.append("Basis matrix cannot be empty")

        if hasattr(config, "inner_config") and config.inner_config is not None:
            try:
                BranchConfigValidator.validate(config.inner_config)
            except ValueError as e:
                errors.append(f"Invalid inner branch config: {str(e)}")

        if errors:
            raise ValueError(f"Orthonormal branch errors: {', '.join(errors)}")