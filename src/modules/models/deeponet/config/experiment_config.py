from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from dataclasses import dataclass, replace
from src.modules.models.deeponet.config import PathConfig
from src.modules.models.deeponet.training_strategies.config import StrategyConfig
from src.modules.models.deeponet.dataset.transform_config import TransformConfig
from src.modules.models.deeponet.config import TrainConfig, DataConfig
if TYPE_CHECKING:
    from src.modules.models.deeponet.config import DeepONetConfig


@dataclass
class ExperimentConfig:
    """Aggregate configuration for the experiment."""
    problem: str
    dataset_version: str
    experiment_version: str
    device: str | torch.device
    precision: str
    model: DeepONetConfig
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
    
    def get_serializable_config(self) -> ExperimentConfig:
        """
        Returns a new, serializable copy of the configuration by removing
        large or non-serializable attributes like POD basis tensors.
        """
        changes = {
            'model': replace(self.model, trunk=replace(self.model.trunk, pod_basis=None)),
            'model': replace(self.model, trunk=replace(self.model.trunk, inner_config=replace(self.model.trunk.inner_config, pod_basis=None) if self.model.trunk.inner_config is not None else None)),
        }

        if self.model.strategy.name == 'pod':
            strategy_changes = replace(self.model.strategy, pod_basis=None if hasattr(self.model.strategy, 'pod_basis') else self.model.strategy)
            model_changes = replace(self.model, 
                                    strategy=strategy_changes,
                                    bias=replace(self.model.bias, precomputed_mean=None),
                                    trunk=replace(self.model.trunk, architecture='precomputed', component_type='pod_trunk'))
            changes['model'] = model_changes

        return replace(self, **changes)