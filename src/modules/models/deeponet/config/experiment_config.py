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
    

    def get_serializable_config(self, updated_strategy) -> ExperimentConfig:
        """
        Returns a new, serializable copy of the configuration by removing
        large or non-serializable attributes like POD basis tensors,
        and ensuring the model configuration is correct for the saved state.
        """
        
        prev_config_copy = deepcopy(self)
        prev_model_config = prev_config_copy.model
        prev_strategy_config = prev_config_copy.strategy
        
        if prev_strategy_config.name == 'pod':
            updated_trunk_config = replace(prev_config_copy.model.trunk, pod_basis=None)

            if updated_trunk_config.inner_config is not None:
                updated_inner_config = replace(updated_trunk_config.inner_config, pod_basis=None)
                updated_trunk_config = replace(updated_trunk_config, inner_config=updated_inner_config)
                
            # modifying this inplace
            prev_config_copy.strategy = replace(prev_strategy_config, pod_basis=None)
            prev_model_config_strategy_clean = replace(prev_config_copy.model.strategy, pod_basis=None)
            updated_bias_config = replace(prev_config_copy.model.bias, precomputed_mean=None)
            
            pod_updated_model_config = replace(
                prev_config_copy.model,
                trunk=updated_trunk_config,
                strategy=prev_model_config_strategy_clean,
                bias=updated_bias_config
            )
        
        elif prev_strategy_config.name == 'two_step':
            
            if hasattr(updated_strategy, 'final_trunk_config') and hasattr(updated_strategy, 'final_branch_config'):
                updated_strategy.final_branch_config.inner_config.output_dim = updated_strategy.final_branch_config.output_dim
                updated_strategy.final_trunk_config.inner_config.output_dim = updated_strategy.final_trunk_config.output_dim
                new_model_config = replace(
                    prev_model_config,
                    trunk=updated_strategy.final_trunk_config,
                    branch=updated_strategy.final_branch_config
                )
            else:
                raise AttributeError("New component config was not found.")

            new_trunk_config = new_model_config.trunk
            new_branch_config = new_model_config.branch
            
            if new_trunk_config.pod_basis is not None:
                new_trunk_config = replace(new_trunk_config, pod_basis=None)
            
            if hasattr(new_trunk_config.inner_config, 'pod_basis'):
                clean_inner_config = replace(new_trunk_config.inner_config, pod_basis=None)
            
            if new_trunk_config.inner_config is not None:
                new_trunk_config = replace(new_trunk_config, inner_config=clean_inner_config)

            two_step_updated_model_config = replace(prev_model_config, trunk=new_trunk_config, branch=new_branch_config)

        strategy_dict = asdict(prev_strategy_config)
        updated_strategy_config = type(prev_strategy_config)(**strategy_dict)
        if hasattr(updated_strategy_config, 'pod_basis'):
            updated_strategy_config = replace(updated_strategy_config, pod_basis=None)
        if hasattr(updated_strategy_config, 'pod_mean'):
            updated_strategy_config = replace(updated_strategy_config, pod_mean=None)

        try:
            del strategy_dict['pod_basis']
            del strategy_dict['pod_mean']
        except KeyError:
            pass

        updated_model_config = pod_updated_model_config if prev_strategy_config.name == 'pod' else two_step_updated_model_config

        print(updated_model_config.branch.output_dim)
        print(updated_model_config.trunk.output_dim)
        # print(updated_strategy.final_branch_config.inner_config.output_dim)
        # print(updated_strategy.final_trunk_config.inner_config.output_dim)
        final_config = replace(prev_config_copy,
                            model=updated_model_config,
                            strategy=updated_strategy_config)
        return final_config