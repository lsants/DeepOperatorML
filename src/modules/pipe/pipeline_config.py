from __future__ import annotations
import os
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from dataclasses import dataclass
from ..model.nn.activation_functions.activation_fns import ACTIVATION_MAP
from ..data_processing.data_augmentation.feature_expansions import FeatureExpansionConfig
from ..model.components.output_handler.config import OutputConfig
from ..model.components.rescaling.config import RescalingConfig
from ..model.config import ModelConfig
from ..data_processing.config import ComponentTransformConfig, TransformConfig
# from ..model.output_handler.config import OutputConfig
from ..model.components.branch.config import BranchConfig
from ..model.components.trunk.config import TrunkConfig
# from ..model.output_handler.rescaling import Rescaling
from ..model.training_strategies.base import StrategyConfig
from ..model.optimization.optimizers.config import OptimizerConfig, OptimizerPhaseConfig

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
            raw_outputs_path = exp_cfg['output_path'],
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
    optimizers: OptimizerConfig  # Phase-keyed
    loss_function: str
    rescaling: dict
    global_optimizer_schedule: list[OptimizerPhaseConfig]

    @classmethod
    def from_config_files(cls, exp_cfg_path: str, train_cfg_path: str, data_cfg: DataConfig):
        # Load YAML files
        with open(exp_cfg_path) as f:
            exp_cfg = yaml.safe_load(f)
        with open(train_cfg_path) as f:
            train_cfg = yaml.safe_load(f)

        branch_config = BranchConfig(**train_cfg["branch"])
        branch_config.activation = ACTIVATION_MAP[branch_config.activation.lower()]
        branch_config.input_dim = data_cfg.shapes[data_cfg.features[0]][1]
        branch_config.output_dim = train_cfg["num_basis_functions"]
        
        trunk_config = TrunkConfig(**train_cfg["trunk"])
        trunk_config.activation = ACTIVATION_MAP[trunk_config.activation.lower()]
        trunk_config.input_dim = data_cfg.shapes[data_cfg.features[1]][1]
        trunk_config.output_dim = train_cfg["num_basis_functions"]
        output_config = OutputConfig(
            handler_type=train_cfg["output_handling"],
            num_channels=len(data_cfg.targets)
        )
        rescaling_config = RescalingConfig(
            num_basis_functions=train_cfg["num_basis_functions"],
            exponent=train_cfg["rescaling"]["exponent"],
        )
        strategy_config = StrategyConfig(name=train_cfg['training_strategy'],
                                        basis_functions=train_cfg["num_basis_functions"],
                                        num_pod_modes=train_cfg["num_pod_modes"],
                                        # pod_basis=train_cfg["pod_basis"],
                                        two_step_branch_epochs=train_cfg["branch_epochs"],
                                        two_step_trunk_epochs=train_cfg["trunk_epochs"],
                                        decomposition_type=train_cfg["decomposition_type"],
                                        global_schedule=train_cfg["global_optimizer_schedule"],
                                        two_step_phase_optimizers=train_cfg["two_step_optimizer_schedule"]
                                        )

        # Build sub-configurations
        model_config = ModelConfig(
            branch=branch_config,
            trunk=trunk_config,
            output=output_config,
            rescaling=rescaling_config,
            strategy=strategy_config
        )
        branch_normalization = train_cfg["transforms"]["branch"]['normalization']
        branch_expansions = train_cfg["transforms"]["branch"]['feature_expansion']
        trunk_normalization = train_cfg["transforms"]["trunk"]['normalization']
        trunk_expansions = train_cfg["transforms"]["trunk"]['feature_expansion']

        transform_config = TransformConfig(branch=ComponentTransformConfig(normalization=branch_normalization \
                                                                           if branch_normalization else None,
                                                                            feature_expansion=FeatureExpansionConfig(type=branch_expansions['type'],
                                                                                                                     size=branch_expansions['size'],
                                                                                                                     original_dim=branch_config.input_dim) \
                                                                                                                        if branch_expansions else None,
                            ),                                  
                                         trunk=ComponentTransformConfig(normalization=trunk_normalization \
                                                                        if trunk_normalization else None,
                                                                            feature_expansion=FeatureExpansionConfig(type=trunk_expansions['type'],
                                                                                                                     size=trunk_expansions['size'],
                                                                                                                      original_dim=trunk_config.input_dim) \
                                                                                                                        if trunk_expansions else None,
                            )
        )
        transform_config.device = exp_cfg["device"]
        transform_config.dtype = exp_cfg["precision"]

        global_optimizers = [
            OptimizerPhaseConfig(**params)
            for params in train_cfg['global_optimizer_schedule']
        ]   
        phase_optimizers = {
            phase: [OptimizerPhaseConfig(**p) for p in phases]
            for phase, phases in train_cfg.get("phase_optimizer_schedule", {}).items()
        }

        optimizer_config = OptimizerConfig(
            global_schedule=global_optimizers,
            phase_specific=phase_optimizers
        )

        return cls(
            precision=exp_cfg["precision"],
            device=exp_cfg["device"],
            branch_batch_size=train_cfg["branch_batch_size"],   
            trunk_batch_size=train_cfg["trunk_batch_size"],   
            model=model_config,
            transforms=transform_config,
            strategy=strategy_config,
            optimizers=optimizer_config,
            global_optimizer_schedule=global_optimizers,
            loss_function = train_cfg['loss_function'],
            rescaling=train_cfg['rescaling']
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
    optimizers: OptimizerConfig

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
        processed_outputs_path = Path(data_cfg.raw_outputs_path)
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