from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..data_processing.config import TransformConfig
from ..model.config import ModelConfig
from ..pipe.pipeline_config import DataConfig, ExperimentConfig, TestConfig, TrainConfig


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
        if train_cfg.transforms.target_normalization is not None:
            raise ValueError("POD requires unnormalized outputs")


def validate_normalization(
    transform_cfg: TransformConfig,
    data_cfg: DataConfig
) -> None:
    """Validate required scaler parameters exist."""
    # Input normalization
    if transform_cfg.branch.normalization == 'standardize' or \
            transform_cfg.trunk.normalization == 'standardize':
        if f'{data_cfg.features[0]}_mean' not in data_cfg.scalers:
            raise ValueError(
                f"Missing '{data_cfg.features[0]}_mean' in scalers")
        if f'{data_cfg.features[0]}_std' not in data_cfg.scalers:
            raise ValueError(
                f"Missing '{data_cfg.features[0]}_std' in scalers")
        if f'{data_cfg.features[1]}_mean' not in data_cfg.scalers:
            raise ValueError(
                f"Missing '{data_cfg.features[1]}_std' in scalers")
        if f'{data_cfg.features[1]}_std' not in data_cfg.scalers:
            raise ValueError(
                f"Missing '{data_cfg.features[1]}_std' in scalers")

    # Output normalization
    if transform_cfg.target_normalization == 'minmax_0_1' or \
            transform_cfg.target_normalization == 'minmax_-1_1':
        if f'{data_cfg.targets[0]}_min' not in data_cfg.scalers:
            raise ValueError(f"Missing '{data_cfg.targets[0]}_min' in scalers")
        if f'{data_cfg.targets[0]}_max' not in data_cfg.scalers:
            raise ValueError(f"Missing '{data_cfg.targets[0]}_min' in scalers")

    if transform_cfg.target_normalization == 'standardize':
        if f'{data_cfg.targets[0]}_mean' not in data_cfg.scalers:
            raise ValueError(
                f"Missing '{data_cfg.targets[0]}_mean' in scalers")
        if f'{data_cfg.targets[0]}_std' not in data_cfg.scalers:
            raise ValueError(f"Missing '{data_cfg.targets[0]}_std' in scalers")


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


def validate_test(test_cfg: TestConfig, exp_cfg: ExperimentConfig) -> None:
    if test_cfg.experiment_version != exp_cfg.experiment_version:
        raise ValueError(
            "Incompatible data configs. Experiment version on Test and Experiment configs don't match.")
