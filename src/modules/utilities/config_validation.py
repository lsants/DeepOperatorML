from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..data_processing.config import TransformConfig
from ..model.config import ModelConfig
from ..pipe.pipeline_config import DataConfig, TrainConfig

# src/modules/utilities/validation.py
def validate_data_config(cfg: DataConfig):
    # Validate split indices match data shapes
    for split in ['train', 'val', 'test']:
        if split not in cfg.split_indices:
            raise ValueError(f"Missing {split} indices")
        
        idx = cfg.split_indices[split]
        for feature in cfg.features + cfg.targets:
            if cfg.shapes[feature][0] != len(idx):
                raise ValueError(
                    f"Split {split} indices don't match {feature} shape"
                )

def validate_train_config(cfg: TrainConfig):
    # Validate device/precision compatibility
    if cfg.device == "cpu" and cfg.precision == "float16":
        raise ValueError("float16 requires CUDA device")
    
    # Validate strategy exists
    valid_strategies = ["pod", "two_step", "standard"]
    if cfg.strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy: {cfg.strategy}")

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
    if train_cfg.strategy.name == 'pod':
        if train_cfg.transforms.output_normalization != 'none':
            raise ValueError("POD requires unnormalized outputs")
        

def validate_normalization(
    transform_cfg: TransformConfig,
    data_cfg: DataConfig
) -> None:
    """Validate required scaler parameters exist."""
    # Input normalization
    if transform_cfg.input_normalization == 'standard':
        if 'input_mean' not in data_cfg.scalers:
            raise ValueError("Missing input_mean in scalers")
        if 'input_std' not in data_cfg.scalers:
            raise ValueError("Missing input_std in scalers")

    # Output normalization
    if transform_cfg.output_normalization == 'minmax':
        if 'output_min' not in data_cfg.scalers:
            raise ValueError("Missing output_min in scalers")
        if 'output_max' not in data_cfg.scalers:
            raise ValueError("Missing output_max in scalers")
        
        
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
    
    # Calculate expanded dimensions
    branch_expansion = transform_cfg.branch_feature_expansion
    trunk_expansion = transform_cfg.trunk_feature_expansion
    
    # Branch dimension validation
    if transform_cfg.branch_feature_expansion:
        branch_degree = transform_cfg.branch_expansion_degree
        expected_branch_dim = original_branch_dim * branch_degree
        if model_cfg.branch.input_dim != expected_branch_dim:
            raise ValueError(
                f"Branch input_dim {data_cfg.} "
                f"≠ {original_branch_dim} * {branch_degree} = {expected_branch_dim}"
            )
    else:
        if model_cfg.branch.input_dim != original_branch_dim:
            raise ValueError(
                f"Branch input_dim {model_cfg.branch.input_dim} "
                f"≠ original {original_branch_dim}"
            )

    # Trunk dimension validation        
    if transform_cfg.trunk_feature_expansion:
        trunk_degree = transform_cfg.trunk_expansion_degree
        expected_trunk_dim = original_trunk_dim * trunk_degree
        if model_cfg.trunk.input_dim != expected_trunk_dim:
            raise ValueError(
                f"Trunk input_dim {model_cfg.trunk.input_dim} "
                f"≠ {original_trunk_dim} * {trunk_degree} = {expected_trunk_dim}"
            )
    else:
        if model_cfg.trunk.input_dim != original_trunk_dim:
            raise ValueError(
                f"Trunk input_dim {model_cfg.trunk.input_dim} "
                f"≠ original {original_trunk_dim}"
            )