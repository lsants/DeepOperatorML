from __future__ import annotations
from dataclasses import dataclass
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Pure dataset-related configuration"""
    problem: str                            # From CLI
    dataset_version: str                    # From experiment.yaml
    dataset_path: Path                      # Constructed from problem + version
    raw_outputs_path: Path
    raw_data_path: Path                     # From metadata.yaml
    raw_metadata_path: Path                 # From metadata.yaml
    split_ratios: list[str]                 # From metadata.yaml
    features: list[str]                     # From metadata.yaml
    input_functions: list[str]              # From metadata.yaml
    coordinates: list[str]                  # From metadata.yaml
    targets: list[str]                      # From metadata.yaml
    targets_labels: list[str]               # From metadata.yaml
    shapes: dict[str, list[int]]            # From metadata.yaml
    data: dict[str, np.ndarray]             # From data.npz
    split_indices: dict[str, np.ndarray]    # From split_indices.npz
    scalers: dict[str, np.ndarray]          # From scalers.npz
    pod_data: dict[str, np.ndarray]         # From pod_data.npz

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
            split_ratios=metadata['split_ratios'],
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
            split_ratios=metadata['split_ratios'],
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