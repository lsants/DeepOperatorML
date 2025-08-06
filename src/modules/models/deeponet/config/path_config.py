from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from src.modules.models.deeponet.config import DataConfig

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