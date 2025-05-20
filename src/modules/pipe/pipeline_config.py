from __future__ import annotations
import os
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass
from ..model.config import ModelConfig
from ..data_processing.config import TransformConfig
from ..model.components.config import ComponentConfig
from ..model.training_strategies.config import StrategyConfig
from ..model.optimization.optimizers.config import OptimizerConfig
from datetime import datetime

@dataclass
class DataConfig:
    """Pure dataset-related configuration"""
    problem: str               # From CLI
    dataset_version: str       # From experiment.yaml
    dataset_path: Path         # Constructed from problem + version
    raw_outputs_path: Path
    features: list[str]        # From metadata.yaml
    targets: list[str]         # From metadata.yaml
    shapes: Dict[str, list]    # From metadata.yaml
    data: Dict[str, np.ndarray]# From data.npz
    split_indices: Dict[str, np.ndarray]  # From split_indices.npz
    scalers: Dict[str, np.ndarray]        # From scalers.npz

    @classmethod
    def from_experiment_config(cls, problem: str, exp_cfg: Dict):
        dataset_path = Path(
            f"data/processed/{problem}_{exp_cfg['dataset_version']}"
        )
        # Validate critical files
        required_files = ['metadata.yaml', 'data.npz', 
                         'split_indices.npz', 'scalers.npz']
        for f in required_files:
            if not (dataset_path / f).exists():
                raise FileNotFoundError(f"Missing {f} in {dataset_path}")
        
        # Load metadata
        with open(dataset_path / "metadata.yaml") as f:
            metadata = yaml.safe_load(f)
        
        return cls(
            problem=problem,
            dataset_version=exp_cfg['dataset_version'],
            dataset_path=dataset_path,
            raw_outputs_path = exp_cfg['outputs_path'],
            features=metadata['FEATURES'],
            targets=metadata['TARGETS'],
            shapes=metadata['SHAPES'],
            data=dict(np.load(dataset_path / "data.npz")),
            split_indices=dict(np.load(dataset_path / "split_indices.npz")),
            scalers=dict(np.load(dataset_path / "scalers.npz"))
        )

@dataclass
class TrainConfig:
    """Aggregate training configuration."""
    precision: str
    device: str
    branch_batch_size: int
    trunk_batch_size: int
    model: ModelConfig
    transforms: TransformConfig
    strategy: StrategyConfig
    optimizers: Dict[str, OptimizerConfig]  # Phase-keyed

    @classmethod
    def from_config_files(cls, exp_cfg_path: str, train_cfg_path: str):
        # Load YAML files
        with open(exp_cfg_path) as f:
            exp_cfg = yaml.safe_load(f)
        with open(train_cfg_path) as f:
            train_cfg = yaml.safe_load(f)


        # Build sub-configurations
        model_config = ModelConfig(
            branch=ComponentConfig(**train_cfg.model.branch),
            trunk=ComponentConfig(**train_cfg.model.trunk),
            output_handling=train_cfg.model.output_handling,
            basis_functions=train_cfg.model.basis_functions,
        )
        transform_config = TransformConfig(**train_cfg["transforms"])
        strategy_config = StrategyConfig(
            name=train_cfg["strategy"]["name"],
            basis_functions=train_cfg["strategy"].basis_functions,
        )
        optimizers = {
            phase: OptimizerConfig(**params)
            for phase, params in train_cfg["optimizers"].items()
        }

        return cls(
            precision=exp_cfg["precision"],
            device=exp_cfg["device"],
            branch_batch_size=train_cfg["branch_batch_size"],   
            trunk_batch_size=train_cfg["trunk_batch_size"],   
            model=model_config,
            transforms=transform_config,
            strategy=strategy_config,
            optimizers=optimizers
        )

@dataclass
class ValidatedConfig:
    data: 'DataConfig'
    training: 'TrainConfig'

class ConfigValidator:
    @staticmethod
    def validate(data_params: Dict[str, Any], 
                train_params: Dict[str, Any]) -> ValidatedConfig:
        # Cross-config validation
        if (train_params['OUTPUT_HANDLING'] == 'split_outputs' and 
            len(data_params['TARGETS']) < 2):
            raise ValueError("Split output handling requires multiple targets")
        
        # Device/precision compatibility
        if (train_params['DEVICE'] == 'cpu' and 
            train_params['PRECISION'] == 'float16'):
            raise ValueError("Float16 precision requires CUDA device")
        
        return ValidatedConfig(
            data=DataConfig(**data_params),
            training=TrainConfig(**train_params)
        )
    
@dataclass
class TestConfig:
    """Aggregate testing configuration."""
    precision: str
    device: str
    output_path: Path
    experiment_version: str
    @classmethod
    def from_config_files(cls, test_cfg_path: str):
        # Load YAML files
        with open(test_cfg_path) as f:
            test_cfg = yaml.safe_load(f)

@dataclass
class ExperimentConfig:
    """Aggregate configuration for the experiment."""
    problem: str
    dataset_version: str
    device: str
    precision: str
    model: ModelConfig
    transforms: TransformConfig
    strategy: StrategyConfig
    optimizers: Dict[str, OptimizerConfig]

    @classmethod
    def from_dataclasses(cls, data_cfg: DataConfig, train_cfg: TrainConfig):
        return cls(
            problem=data_cfg.problem,
            dataset_version=data_cfg.dataset_version,
            device=train_cfg.device,
            precision=train_cfg.precision,
            model=train_cfg.model,
            transforms=train_cfg.transforms,
            strategy=train_cfg.strategy,
            optimizers=train_cfg.optimizers
        )
    
@dataclass
class PathConfig:
    outputs_path: Path
    checkpoints_path: Path
    auxiliary_data_path: Path
    model_info_path: Path
    dataset_indices_path: Path
    norm_params_path: Path
    metrics_path: Path
    plots_path: Path

    @classmethod
    def from_data_config(cls, data_cfg: DataConfig):
        processed_outputs_path = data_cfg.raw_outputs_path
        experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return cls(
            outputs_path=processed_outputs_path / data_cfg.problem / experiment_name,
            checkpoints_path=processed_outputs_path / "checkpoints",
            auxiliary_data_path=processed_outputs_path / 'aux',
            model_info_path=processed_outputs_path / 'config.yaml',
            dataset_indices_path=processed_outputs_path / 'aux' / 'split_indices.yaml',
            norm_params_path=processed_outputs_path / 'aux' / 'norm_params.yaml',
            metrics_path=processed_outputs_path / 'metrics',
            plots_path=processed_outputs_path / 'plots'
        )
    
    @classmethod
    def create_directories(cls, config: PathConfig):
        paths = [
            config.outputs_path,
            config.checkpoints_path,
            config.auxiliary_data_path,
            config.metrics_path,
            config.plots_path
        ]
        for path in paths:
            os.makedirs(path, exist_ok=True)