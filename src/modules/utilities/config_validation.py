from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..data_processing.config import TransformConfig
from ..model.config import ModelConfig
from ..pipe.pipeline_config import DataConfig, TrainConfig

def validate_train_config(cfg: TrainConfig):
    # Validate device/precision compatibility
    if cfg.device == "cpu" and cfg.precision == "float16":
        raise ValueError("float16 requires CUDA device")
    
    # Validate strategy exists
    valid_strategies = ["pod", "two_step", "vanilla"]
    if cfg.model.strategy.name not in valid_strategies:
        raise ValueError(f"Invalid strategy: {cfg.model.strategy.name}")

def validate_config_compatibility(
    data_cfg: DataConfig,
    train_cfg: TrainConfig
) -> None:
    """Ensure transforms align with other configs."""
    # Feature expansion vs model input dimensions
    validate_feature_expansion(
        transform_cfg=train_cfg.transforms,
        model_cfg=train_cfg.model,
        data_cfg=data_cfg
    )
    
    # Normalization vs scaler parameters
    validate_normalization(
        transform_cfg=train_cfg.transforms, 
        data_cfg=data_cfg
        )
    
    # Strategy-specific rules
    if train_cfg.model.strategy.name == 'pod':
        if train_cfg.transforms.target_normalization != 'none':
            raise ValueError("POD requires unnormalized outputs")
        

def validate_normalization(
    transform_cfg: TransformConfig,
    data_cfg: DataConfig
) -> None:
    """Validate required scaler parameters exist."""
    # Input normalization
    if transform_cfg.branch.normalization == 'standardize' or \
    transform_cfg.trunk.normalization == 'standardize':
        if 'input_mean' not in data_cfg.scalers:
            raise ValueError("Missing input_mean in scalers")
        if 'input_std' not in data_cfg.scalers:
            raise ValueError("Missing input_std in scalers")

    # Output normalization
    if transform_cfg.target_normalization == 'minmax':
        if 'target_min' not in data_cfg.scalers:
            raise ValueError("Missing target_min in scalers")
        if 'target_max' not in data_cfg.scalers:
            raise ValueError("Missing target_max in scalers")
        
        
def validate_feature_expansion(transform_cfg: TransformConfig, 
                              model_cfg: ModelConfig, 
                              data_cfg: DataConfig) -> None:
    """
    Validate transformed dimensions match model expectations
    for both branch and trunk components.
    """
    # Get base dimensions from data config
    features = data_cfg.features
    original_branch_dim = data_cfg.shapes[features[0]][1]
    original_trunk_dim = data_cfg.shapes[features[1]][1]


    if model_cfg.branch.input_dim != original_branch_dim:
        raise ValueError(
            f"Branch input_dim {model_cfg.branch.input_dim} "
            f"≠ original {original_branch_dim}"
        )


    if model_cfg.trunk.input_dim != original_trunk_dim:
        raise ValueError(
            f"Trunk input_dim {model_cfg.trunk.input_dim} "
            f"≠ original {original_trunk_dim}"
        )
        
