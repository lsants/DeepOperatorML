from __future__ import annotations
import dataclasses
import os
import torch
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Any
from datetime import datetime
from dataclasses import dataclass, replace
from ..model.nn.activation_functions.activation_fns import ACTIVATION_MAP
from ..model.components.output_handler.config import OutputConfig
from ..model.components.rescaling.config import RescalingConfig
from ..model.config import ModelConfig
from ..data_processing.config import TransformConfig
from ..model.components.bias.config import BiasConfig
from ..model.components.branch.config import BranchConfig
from ..model.components.trunk.config import TrunkConfig
from ..model.optimization.optimizers.config import OptimizerSpec
from ..model.training_strategies.config import StrategyConfig


logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Pure dataset-related configuration"""
    problem: str               # From CLI
    dataset_version: str       # From experiment.yaml
    dataset_path: Path         # Constructed from problem + version
    raw_outputs_path: Path
    raw_data_path: Path        # From metadata.yaml
    raw_metadata_path: Path    # From metadata.yaml
    features: list[str]        # From metadata.yaml
    input_functions: list[str]        # From metadata.yaml
    coordinates: list[str]        # From metadata.yaml
    targets: list[str]         # From metadata.yaml
    targets_labels: list[str]  # From metadata.yaml
    shapes: dict[str, list[int]]    # From metadata.yaml
    data: dict[str, np.ndarray]  # From data.npz
    split_indices: dict[str, np.ndarray]  # From split_indices.npz
    scalers: dict[str, np.ndarray]        # From scalers.npz
    pod_data: dict[str, np.ndarray]      # From pod_data.npz

    @classmethod
    def from_experiment_config(cls, problem: str, exp_cfg: dict[str, Any]):
        dataset_path = Path(
            f"data/processed/{problem}/{exp_cfg['dataset_version']}"
        )
        required_files = ['metadata.yaml',
                          'data.npz',
                          'split_indices.npz',
                          'scalers.npz',
                          'pod.npz'
                          ]
        for f in required_files:
            if not (dataset_path / f).exists():
                raise FileNotFoundError(f"Missing {f} in {dataset_path}")

        # Load processed metadata
        with open(dataset_path / "metadata.yaml") as f:
            metadata = yaml.safe_load(f)

        return cls(
            problem=problem,
            dataset_version=exp_cfg['dataset_version'],
            dataset_path=dataset_path,
            raw_outputs_path=exp_cfg['output_path'],
            raw_data_path=metadata['raw_data_source'],
            raw_metadata_path=metadata['raw_metadata_source'],
            features=metadata['features'],
            input_functions=metadata['input_functions'],
            coordinates=metadata['coordinates'],
            targets=metadata['targets'],
            targets_labels=metadata['targets_labels'],
            shapes=metadata['shapes'],
            data=dict(np.load(dataset_path / "data.npz")),
            split_indices=dict(np.load(dataset_path / "split_indices.npz")),
            scalers=dict(np.load(dataset_path / "scalers.npz")),
            pod_data=dict(np.load(dataset_path / "pod.npz"))
        )

    @classmethod
    def from_test_config(cls, problem: str, exp_cfg: dict[str, Any]):
        dataset_path = Path(
            f"data/processed/{problem}/{exp_cfg['dataset_version']}"
        )
        required_files = ['metadata.yaml', 'data.npz',
                          'split_indices.npz', 'scalers.npz', 'pod_data.npz'
                          ]
        for f in required_files:
            if not (dataset_path / f).exists():
                raise FileNotFoundError(f"Missing {f} in {dataset_path}")

        with open(dataset_path / "metadata.yaml") as f:
            metadata = yaml.safe_load(f)

        return cls(
            problem=problem,
            dataset_version=exp_cfg['dataset_version'],
            dataset_path=dataset_path,
            raw_outputs_path=exp_cfg['output_path'],
            raw_data_path=metadata['raw_data_source'],
            raw_metadata_path=metadata['raw_metadata_source'],
            features=metadata['features'],
            input_functions=metadata['input_functions'],
            coordinates=metadata['coordinates'],
            targets=metadata['targets'],
            targets_labels=metadata['targets_labels'],
            shapes=metadata['shapes'],
            data=dict(np.load(dataset_path / "data.npz")),
            split_indices=dict(np.load(dataset_path / "split_indices.npz")),
            scalers=dict(np.load(dataset_path / "scalers.npz")),
            pod_data=dict(np.load(dataset_path / "pod.npz"))
        )


@dataclass
class TrainConfig:
    """Aggregate training configuration."""
    precision: str
    device: str | torch.device
    seed: int
    branch_batch_size: int
    trunk_batch_size: int
    model: ModelConfig
    transforms: TransformConfig
    strategy: dict
    rescaling: dict
    pod_data: dict[str, torch.Tensor]

    @classmethod
    def from_config_files(cls, exp_cfg_path: str, train_cfg_path: str, data_cfg: DataConfig):
        with open(exp_cfg_path) as f:
            exp_cfg = yaml.safe_load(f)
        with open(train_cfg_path) as f:
            train_cfg = yaml.safe_load(f)

        pod_mask = 'multi' if train_cfg['output_handling'] == 'split_outputs' else 'single'

        pod_data = {
            k: torch.tensor(v).to(
                device=exp_cfg["device"],
                dtype=getattr(torch, exp_cfg["precision"])
            )
            for k, v in data_cfg.pod_data.items() if pod_mask in k
        }
        pod_data = {
            'pod_basis': pod_data[f"{pod_mask}_basis"],
            'pod_mean': pod_data[f"{pod_mask}_mean"]
        }

        train_cfg["trunk"]['pod_basis'] = pod_data['pod_basis']

        trunk_config = TrunkConfig.setup_for_training(
            dataclasses.asdict(data_cfg), train_cfg)
        branch_config = BranchConfig.setup_for_training(
            dataclasses.asdict(data_cfg), train_cfg)
        bias_config = BiasConfig.setup_for_training(
            pod_data=pod_data if train_cfg['training_strategy'] == 'pod' else None, data_cfg=dataclasses.asdict(data_cfg))

        branch_config.input_dim = data_cfg.shapes[data_cfg.features[0]][1]
        branch_config.output_dim = trunk_config.output_dim

        output_config = OutputConfig.setup_for_training(
            train_cfg=train_cfg, data_cfg=dataclasses.asdict(data_cfg))
        rescaling_config = RescalingConfig.setup_for_training(train_cfg)
        if train_cfg['training_strategy'] == 'pod':
            rescaling_config.num_basis_functions = branch_config.output_dim // output_config.num_channels  # type: ignore
        one_step_optimizer = [
            OptimizerSpec(**params)
            for params in train_cfg['optimizer_schedule']
        ]
        multi_step_optimizer = {
            phase: [OptimizerSpec(**p) for p in phases]
            for phase, phases in train_cfg.get("two_step_optimizer_schedule", {}).items()
        }

        strategy_config = {
            'name': train_cfg['training_strategy'],
            'error': train_cfg['error'],
            'loss': train_cfg['loss_function'],
            'optimizer_scheduler': one_step_optimizer,
            'two_step_optimizer_schedule': multi_step_optimizer,
            'decomposition_type': train_cfg['decomposition_type'],
            **pod_data
        }

        model_config = ModelConfig(
            branch=branch_config,
            trunk=trunk_config,
            bias=bias_config,
            output=output_config,
            rescaling=rescaling_config,
            strategy=strategy_config  # type: ignore
        )

        device = torch.device(exp_cfg["device"])
        dtype = getattr(torch, exp_cfg["precision"])

        transform_config = TransformConfig.from_train_config(
            branch_transforms=train_cfg["transforms"]["branch"],
            trunk_transforms=train_cfg["transforms"]["trunk"],
            target_transforms=train_cfg["transforms"]["target"],
            device=device,
            dtype=dtype
        )

        return cls(
            precision=dtype,
            device=device,
            seed=train_cfg['seed'],
            branch_batch_size=train_cfg["branch_batch_size"],
            trunk_batch_size=train_cfg["trunk_batch_size"],
            model=model_config,
            transforms=transform_config,
            strategy=strategy_config,
            rescaling=train_cfg['rescaling'],
            pod_data=pod_data
        )


@dataclass
class ValidatedConfig:
    data: 'DataConfig'
    training: 'TrainConfig'


class ConfigValidator:
    @staticmethod
    def validate(data_params: dict[str, Any],
                 train_params: dict[str, Any]) -> ValidatedConfig:
        if (train_params['OUTPUT_HANDLING'] == 'split_outputs' and
                len(data_params['TARGETS']) < 2):
            raise ValueError("Split output handling requires multiple targets")

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
    device: str | torch.device
    processed_data_path: Path
    output_path: Path
    experiment_version: str
    problem: str | None = None
    model: ModelConfig | None = None
    transforms: TransformConfig | None = None
    metric: str | None = None
    checkpoint: dict | None = None
    config: dict[str, Any] | None = None

    @classmethod
    def from_config_files(cls, test_cfg_path: str):
        with open(test_cfg_path) as f:
            test_cfg = yaml.safe_load(f)

        return cls(
            precision=test_cfg['precision'],
            device=test_cfg['device'],
            processed_data_path=Path(test_cfg['processed_data_path']),
            output_path=Path(test_cfg['output_path']),
            experiment_version=test_cfg['experiment_version'],
            config=test_cfg
        )

    def with_experiment_data(self, exp_cfg_dict: dict[str, Any]):
        def _create_transforms_config(cfg_dict: dict[str, Any]) -> TransformConfig:
            return TransformConfig.from_exp_config(
                branch_transforms=cfg_dict["branch"],
                trunk_transforms=cfg_dict["trunk"],
                target_transforms=cfg_dict["target"],
                device=self.device,
                dtype=getattr(torch, self.precision)
            )
        transforms_config = _create_transforms_config(
            exp_cfg_dict['transforms'])
        self.transforms = transforms_config

        def _create_model_config(cfg_dict: dict[str, Any]) -> ModelConfig:
            if self.transforms is None:
                raise ValueError("Transforms config not initialized")
            if self.transforms.branch.feature_expansion is None or self.transforms.trunk.feature_expansion is None:
                raise ValueError("Components config not initialized")
            if self.transforms.branch.original_dim is None or self.transforms.trunk.original_dim is None:
                raise ValueError("Missing original dims for initialization")

            bias_config = BiasConfig.setup_for_inference(
                cfg_dict)
            branch_config = BranchConfig.setup_for_inference(
                cfg_dict, self.transforms)
            trunk_config = TrunkConfig.setup_for_inference(
                cfg_dict, self.transforms)
            if cfg_dict['strategy']['name'] == 'pod':
                # type: ignore
                trunk_config.output_dim = trunk_config.pod_basis_shape[-1]
            branch_config.output_dim = trunk_config.output_dim
            output_config = OutputConfig.setup_for_inference(cfg_dict)
            rescaling_config = RescalingConfig.setup_for_inference(cfg_dict)
            strategy_config = StrategyConfig.setup_for_inference(cfg_dict)

            return ModelConfig(
                branch=branch_config,
                trunk=trunk_config,
                bias=bias_config,
                output=output_config,
                rescaling=rescaling_config,
                strategy=strategy_config
            )

        model_config = _create_model_config(exp_cfg_dict["model"])

        return replace(
            self,
            problem=exp_cfg_dict['problem'],
            model=model_config,
            metric=exp_cfg_dict['strategy']['error']
        )

    def with_checkpoint(self, checkpoint: dict[str, Any]):
        """
        Creates a new TestConfig instance by updating the current one with a checkpoint dictionary.
        """
        return replace(self, checkpoint=checkpoint)


@dataclass
class ExperimentConfig:
    """Aggregate configuration for the experiment."""
    problem: str
    dataset_version: str
    experiment_version: str
    device: str | torch.device
    precision: str
    model: ModelConfig
    transforms: TransformConfig
    strategy: StrategyConfig

    @classmethod
    def from_dataclasses(cls, data_cfg: DataConfig, train_cfg: TrainConfig, path_cfg: PathConfig):

        train_cfg.transforms.branch.original_dim = data_cfg.shapes[data_cfg.features[0]][1]
        train_cfg.transforms.trunk.original_dim = data_cfg.shapes[data_cfg.features[1]][1]

        return cls(
            problem=data_cfg.problem,
            dataset_version=data_cfg.dataset_version,
            experiment_version=path_cfg.experiment_version,
            device=train_cfg.device,
            precision=train_cfg.precision,
            model=train_cfg.model,
            transforms=train_cfg.transforms,
            strategy=train_cfg.model.strategy
        )


@dataclass
class PathConfig:
    experiment_version: str
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
        outputs_path = processed_outputs_path / data_cfg.problem / experiment_name
        return cls(
            experiment_version=experiment_name,
            outputs_path=outputs_path,
            checkpoints_path=outputs_path / "checkpoints",
            auxiliary_data_path=outputs_path / 'aux',
            model_info_path=outputs_path / 'config.yaml',
            dataset_indices_path=outputs_path / 'aux' / 'split_indices.yaml',
            norm_params_path=outputs_path / 'aux' / 'norm_params.yaml',
            metrics_path=outputs_path / 'metrics',
            plots_path=outputs_path / 'plots'
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


def format_exp_cfg(exp_cfg: ExperimentConfig) -> ExperimentConfig:
    exp_cfg.model.trunk.pod_basis = None
    if exp_cfg.model.trunk.inner_config is not None:
        exp_cfg.model.trunk.inner_config.pod_basis = None
    if exp_cfg.model.strategy.name == 'pod':  # type: ignore
        if hasattr(exp_cfg.model.strategy, 'pod_basis'):
            exp_cfg.model.strategy.pod_basis = None  # type: ignore
        exp_cfg.model.bias.precomputed_mean = None
        exp_cfg.model.trunk.architecture = 'precomputed'  # type: ignore
        exp_cfg.model.trunk.component_type = 'pod_trunk'  # type: ignore
    return exp_cfg
