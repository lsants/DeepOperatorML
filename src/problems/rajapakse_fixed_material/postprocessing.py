from __future__ import annotations
import yaml
import logging
import numpy as np
from pathlib import Path
from src.modules.pipe.pipeline_config import DataConfig, TestConfig
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
    delta = raw_data['delta']
    input_functions = {data_cfg.input_functions[0]: delta}
    return input_functions


def get_coordinates(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw_data = np.load(data_cfg.raw_data_path)
    r = raw_data['r']
    z = raw_data['z']
    coordinates = {'r': r, 'z': z}
    return coordinates

def format_target(displacements: np.ndarray, data_cfg: DataConfig) -> np.ndarray:
    with open(data_cfg.raw_metadata_path, 'r') as file:
        raw_metadata = yaml.safe_load(file)
    displacements = displacements.reshape(
        -1,
        raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][1],
        raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][2],
        data_cfg.shapes[data_cfg.targets[0]][-1],
    )
    displacements_flipped = np.flip(displacements, axis=1)
    displacements_full = np.concatenate(
        (displacements_flipped, displacements), axis=1).transpose(0, 3, 1, 2)
    return displacements_full

def reshape_coefficients(branch_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    return branch_out.reshape(
        -1,
        data_cfg.shapes[data_cfg.targets[0]][-1],
        test_cfg.model.rescaling.num_basis_functions,  # type: ignore
    )


def reshape_basis(trunk_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    with open(data_cfg.raw_metadata_path, 'r') as file:
        raw_metadata = yaml.safe_load(file)
    basis = trunk_out.T.reshape(
        test_cfg.model.rescaling.num_basis_functions,  # type: ignore
        -1,
        raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][1],
        raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][2],
    )
    basis_flipped = np.flip(basis, axis=2)
    basis_full = np.concatenate(
        (basis_flipped, basis), axis=2)
    return basis_full

def format_bias(bias: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    if test_cfg.model.strategy.name != 'pod':  # type: ignore
        return bias
    else:
        with open(data_cfg.raw_metadata_path, 'r') as file:
            raw_metadata = yaml.safe_load(file)
        bias = bias.T.reshape(
            -1,
            raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][1],
            raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][2],
        )
        bias_flipped = np.flip(bias, axis=1)
        bias_full = np.concatenate((bias_flipped, bias), axis=1)
        return bias_full
