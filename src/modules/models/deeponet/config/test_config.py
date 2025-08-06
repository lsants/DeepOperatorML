from __future__ import annotations
import torch
import yaml
from pathlib import Path
from typing import Any
from dataclasses import dataclass, replace
from src.modules.models.deeponet.config.deeponet_config import DeepONetConfig
from src.modules.models.deeponet.dataset.transform_config import TransformConfig
from src.modules.models.deeponet.components.output_handler.config import OutputConfig
from src.modules.models.deeponet.components.rescaling.config import RescalingConfig
from src.modules.models.deeponet.components.bias.config import BiasConfig
from src.modules.models.deeponet.components.branch.config import BranchConfig
from src.modules.models.deeponet.components.trunk.config import TrunkConfig
from src.modules.models.deeponet.training_strategies.config import StrategyConfig

@dataclass
class TestConfig:
    """Aggregate testing configuration."""
    precision: str
    device: str | torch.device
    processed_data_path: Path
    output_path: Path
    experiment_version: str
    problem: str | None = None
    model: DeepONetConfig | None = None
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

        def _create_model_config(cfg_dict: dict[str, Any]) -> DeepONetConfig:
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
                trunk_config.output_dim = trunk_config.pod_basis_shape[-1] # type: ignore
            branch_config.output_dim = trunk_config.output_dim
            output_config = OutputConfig.setup_for_inference(cfg_dict)
            rescaling_config = RescalingConfig.setup_for_inference(cfg_dict)
            strategy_config = StrategyConfig.setup_for_inference(cfg_dict)

            return DeepONetConfig(
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