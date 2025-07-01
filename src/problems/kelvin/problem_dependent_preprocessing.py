from __future__ import annotations
import os
import yaml
import logging
import numpy as np
import numpy.typing as npt
from typing import Any, Iterable
from pathlib import Path
import preprocess_data
from src.modules.data_processing import preprocessing_helper as ppr
from src.modules.model.components import trunk

logger = logging.getLogger(__name__)


def input_function_encoding(input_funcs: Iterable[Iterable | npt.NDArray], encoding=None) -> npt.NDArray:
    return np.column_stack(tup=input_funcs)  # type: ignore


def format_to_don(*coords: Iterable[npt.ArrayLike]) -> npt.NDArray:
    if len(coords) == 1 and isinstance(coords[0], Iterable):
        coords = coords[0]  # type: ignore

    meshes = np.meshgrid(*coords, indexing='ij')  # type: ignore

    axes = [m.flatten() for m in meshes]
    data = np.column_stack(axes)
    return data


def preprocess_raw_data(raw_npz_filename: str,
                        input_function_keys: list[str],
                        coordinate_keys: list[str],
                        processed_dataset_keys: dict[str, list[str]],
                        load_direction: str) -> dict[str, npt.NDArray]:
    """
    Processed data from an npz file and groups the input functions and coordinates into arrays
    called, named according to given labels, suitable for creating the PyTorch dataset for the Kelvin problem.

    The function assumes that:
      - The input functions (sensors) are stored under keys given by input_function_keys (num_sensors, num_input_functions).
      - The coordinate arrays (for the trunk) are stored under keys given by coordinate_keys.
        A meshgrid is created and then flattened to yield a 2D array of shape
        (num_coordinate_points, num_coordinate_dimensions).
      - Operator output under the key 'g_u'.

    Args:
        npz_filename (str): Path to the .npz file.
        input_function_keys (list of str): List of keys for sensor (input function) arrays.
        coordinate_keys (list of str): List of keys for coordinate arrays.
        processed_dataset_keys (dict): Map of {'FEATURES': usually ['xb', 'xt], 'TARGETS': usually ['g_u']}.

    Returns:
        dict: A dictionary with the example keys:
            - 'xb': A 2D numpy array of shape (num_sensor_points, num_sensor_dimensions).
            - 'xt': A 2D numpy array of shape (num_coordinate_points, num_coordinate_dimensions).
            - 'g_u': The operator output array.
    """

    data = np.load(file=raw_npz_filename,
                   allow_pickle=True)  # 'label': NDArray

    input_funcs = [data[key] for key in input_function_keys]
    coords = [data[name] for name in coordinate_keys]

    branch_input = input_function_encoding(input_funcs=input_funcs)
    trunk_input = format_to_don(coords)

    features = {processed_dataset_keys['features'][0]: branch_input,
                processed_dataset_keys['features'][1]: trunk_input,
                }
    num_channels = processed_dataset_keys

    load_index = coordinate_keys.index(load_direction)

    processed_data = features

    for i in processed_dataset_keys['targets']:
        if i in data:
            processed_data[i] = data[i]
            num_samples = len(branch_input)
            num_coords = len(trunk_input)
            processed_data[i] = processed_data[i].reshape(
                num_samples, num_coords, -1)
        else:
            raise KeyError(
                f"Operator target '{processed_dataset_keys['targets'][0]}' must be present in the dataset keys")
    return processed_data


def run_preprocessing(problem_settings: dict[str, Any]) -> dict[str, npt.ArrayLike]:
    processed_data = preprocess_raw_data(raw_npz_filename=problem_settings['raw_data_path'],
                                         input_function_keys=problem_settings['input_function_keys'],
                                         coordinate_keys=problem_settings['coordinate_keys'],
                                         processed_dataset_keys=problem_settings['data_labels'],
                                         load_direction=problem_settings['load_direction'])
    return processed_data  # type: ignore
