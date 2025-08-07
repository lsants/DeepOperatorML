from __future__ import annotations
import torch
from copy import deepcopy
from typing import TYPE_CHECKING
from dataclasses import dataclass, replace, asdict
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
    

    def get_serializable_config(self, strategy) -> ExperimentConfig:
        """
        Returns a new, serializable copy of the configuration by removing
        large or non-serializable attributes like POD basis tensors,
        and ensuring the model configuration is correct for the saved state.
        """
        
        config_copy = deepcopy(self)
        updated_model = config_copy.model
        updated_strategy = config_copy.strategy
        
        if config_copy.strategy.name == 'pod':
            updated_trunk = replace(config_copy.model.trunk, pod_basis=None)
            if updated_trunk.inner_config is not None:
                updated_inner_config = replace(updated_trunk.inner_config, pod_basis=None)
                updated_trunk = replace(updated_trunk, inner_config=updated_inner_config)
                
            updated_strategy = replace(config_copy.strategy, pod_basis=None)
            updated_model_strategy = replace(config_copy.model.strategy, pod_basis=None)
            updated_bias = replace(config_copy.model.bias, precomputed_mean=None)
            
            updated_model = replace(config_copy.model,
                                    trunk=updated_trunk,
                                    strategy=updated_model_strategy,
                                    bias=updated_bias)
        
        elif config_copy.strategy.name == 'two_step':
            # Start with the model from the initial copy
            current_model = config_copy.model
            
            # 1. If there's a final config, create a new model with the final trunk and branch
            if hasattr(strategy, 'final_trunk_config'):
                current_model = replace(
                    current_model,
                    trunk=strategy.final_trunk_config,
                    branch=strategy.final_branch_config
                )
            else:
                raise AttributeError("New component config was not found.")

            # 2. Now, create a new, cleaned trunk from the current model's trunk
            new_trunk = current_model.trunk
            
            # Clean the inner_config first (inside-out)
            if new_trunk.inner_config is not None and hasattr(new_trunk.inner_config, 'pod_basis'):
                if new_trunk.inner_config.pod_basis is not None:
                    cleaned_inner_config = replace(new_trunk.inner_config, pod_basis=None)
                    # Update the new_trunk with the cleaned inner_config
                    new_trunk = replace(new_trunk, inner_config=cleaned_inner_config)
            
            # Clean the trunk's pod_basis
            if new_trunk.pod_basis is not None:
                new_trunk = replace(new_trunk, pod_basis=None)

            # 3. Finally, create the fully updated model with the new, cleaned trunk
            updated_model = replace(current_model, trunk=new_trunk)

        strategy_dict = asdict(updated_strategy)
        try:
            del strategy_dict['pod_basis']
            del strategy_dict['pod_mean']
        except KeyError:
            pass
        
        updated_strategy = type(updated_strategy)(**strategy_dict)

        final_config = replace(config_copy,
                            model=updated_model,
                            strategy=updated_strategy)
        
        return final_config