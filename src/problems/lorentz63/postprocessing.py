from __future__ import annotations
import yaml
import logging
import numpy as np
from pathlib import Path
from src.modules.models.deeponet.config import DataConfig, TestConfig

logger = logging.getLogger(__file__)

def get_output_data(test_cfg: TestConfig) -> dict[str, np.ndarray]:
    if test_cfg.problem is None:
        raise ValueError(f"Problem name must be set in TestConfig.")
    base_dir = Path(__file__).parent.parent.parent.parent
    output_data_path = base_dir / test_cfg.output_path / test_cfg.problem / \
        test_cfg.experiment_version / 'aux' / 'output_data.npz'
    output_data = {i: j for i, j in np.load(output_data_path).items()}
    return output_data

def get_input_functions(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw_data = np.load(data_cfg.raw_data_path)
    r_0 = raw_data['r_0']
    input_functions = {data_cfg.input_functions[0]: r_0}
    return input_functions

def get_coordinates(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw_data = np.load(data_cfg.raw_data_path)
    t = raw_data['t']
    coordinates = {'t': t}
    return coordinates

def format_target(trajectories: np.ndarray, data_cfg: DataConfig) -> np.ndarray:
    return trajectories

def reshape_coefficients(branch_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    return branch_out.reshape(
        int(data_cfg.shapes[data_cfg.targets[0]][0]*data_cfg.split_ratios[-1]),
        -1,
        test_cfg.model.rescaling.embedding_dimension,  # type: ignore
    )

def reshape_basis(trunk_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    with open(data_cfg.raw_metadata_path, 'r') as file:
        raw_metadata = yaml.safe_load(file)
    basis = trunk_out.T.reshape(
        test_cfg.model.rescaling.embedding_dimension,  # type: ignore
        -1,
        raw_metadata["trajectories"][data_cfg.targets[0]]["shape"][1],
    )
    return basis # (embedding_size, n_channels, n_coord)

def format_bias(bias: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    if test_cfg.model.strategy.name != 'pod':  # type: ignore
        return bias
    else:
        with open(data_cfg.raw_metadata_path, 'r') as file:
            raw_metadata = yaml.safe_load(file)
        bias = bias.T.reshape(
            -1,
            raw_metadata["trajectories"][data_cfg.targets[0]]["shape"][-1], # TODO: Fix this so that it works also for pod stacked
            raw_metadata["trajectories"][data_cfg.targets[0]]["shape"][1],
        )
        return bias # (embedding_size, n_channels, n_coord)
