from __future__ import annotations
import torch
import yaml
import dataclasses
from dataclasses import dataclass
from src.modules.models.deeponet.config.data_config import DataConfig
from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig
from src.modules.models.deeponet.dataset.transform_config import TransformConfig
from src.modules.models.tools.optimizers.config import OptimizerSpec
from src.modules.models.deeponet.components.output_handler.config import OutputConfig
from src.modules.models.deeponet.components.rescaling.config import RescalingConfig
from src.modules.models.deeponet.components.bias.config import BiasConfig
from src.modules.models.deeponet.components.branch.config import BranchConfig
from src.modules.models.deeponet.components.trunk.config import TrunkConfig


@dataclass
class TrainConfig:
    """Aggregate training configuration."""
    precision: str
    device: str | torch.device
    seed: int
    branch_batch_size: int
    num_branch_train_samples: int
    trunk_batch_size: int
    model: DeepONetConfig
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

        pod_mask = 'split' if train_cfg['pod'] == 'by_channel' else 'stacked'

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
            pod_data=pod_data if train_cfg['training_strategy'] == 'pod' else None, 
            data_cfg=dataclasses.asdict(data_cfg),
            use_zero_bias=train_cfg["bias"]["use_zero_bias"]
            )

        output_config = OutputConfig.setup_for_training(
            train_cfg=train_cfg, data_cfg=dataclasses.asdict(data_cfg))
        rescaling_config = RescalingConfig.setup_for_training(train_cfg)


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
            'num_branch_train_samples': int(data_cfg.split_ratios[0] * data_cfg.shapes[data_cfg.features[0]][0]),
            **pod_data
        }

        model_config = DeepONetConfig(
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
            num_branch_train_samples=int(data_cfg.split_ratios[0] * data_cfg.shapes[data_cfg.features[0]][0]),
            trunk_batch_size=train_cfg["trunk_batch_size"],
            model=model_config,
            transforms=transform_config,
            strategy=strategy_config,
            rescaling=train_cfg['rescaling'],
            pod_data=pod_data
        )