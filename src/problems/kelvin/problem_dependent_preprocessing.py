from __future__ import annotations
import os
import yaml
import logging
import numpy as np
import numpy.typing as npt
from typing import Any, Iterable
from pathlib import Path
from src.modules.data_processing import preprocessing_helper as ppr

logger = logging.getLogger(__name__)

def input_function_encoding(input_funcs: Iterable[Iterable | npt.NDArray], encoding=None) -> npt.NDArray:
    return np.column_stack(tup=input_funcs) # type: ignore

def format_to_don(*coords: Iterable[npt.ArrayLike]) -> npt.NDArray:
    if len(coords) == 1 and isinstance(coords[0], Iterable):
        coords = coords[0] # type: ignore
    
    meshes = np.meshgrid(*coords, indexing='ij') # type: ignore

    axes = [m.flatten() for m in meshes]
    data = np.column_stack(axes)
    return data

def preprocess_raw_data(raw_npz_filename: str, 
                        input_function_keys: list[str],
                        coordinate_keys: list[str],
                        processed_dataset_keys: dict[str, str],
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

    data = np.load(file=raw_npz_filename, allow_pickle=True) # 'label': NDArray
    
    input_funcs = [data[key] for key in input_function_keys]
    coords = [data[name] for name in coordinate_keys]

    branch_input = input_function_encoding(input_funcs=input_funcs)
    trunk_input = format_to_don(coords)
    
    result = {processed_dataset_keys['FEATURES'][0]: branch_input, 
              processed_dataset_keys['FEATURES'][1]: trunk_input}
    
    load_index = coordinate_keys.index(load_direction)

    if processed_dataset_keys['TARGETS'][0] in data:
        result[processed_dataset_keys['TARGETS'][0]] = data[processed_dataset_keys['TARGETS'][0]][..., load_index]
        num_samples = len(branch_input)
        result[processed_dataset_keys['TARGETS'][0]] = result[processed_dataset_keys['TARGETS'][0]].reshape(num_samples, -1)
    else:
        raise KeyError(f"Operator target '{processed_dataset_keys['TARGETS'][0]}' must be present in the dataset keys")
    return result

def run_preprocessing(problem_settings: dict[str, Any]) -> dict[str, npt.ArrayLike]:
    processed_data = preprocess_raw_data(raw_npz_filename=problem_settings['RAW_DATA_PATH'],
                                         input_function_keys=problem_settings['INPUT_FUNCTION_KEYS'],
                                         coordinate_keys=problem_settings['COORDINATE_KEYS'],
                                         processed_dataset_keys=problem_settings['DATA_LABELS'],
                                         load_direction=problem_settings['DIRECTION'])
    return processed_data # type: ignore

if __name__ == '__main__':
    problem = 'kelvin'
    problem_path = "./configs/problems/" + 'kelvin' + '/' + "config_problem.yaml"
    with open(file=problem_path) as f:
        problem_cfg = yaml.safe_load(stream=f)
    run_preprocessing(problem_settings=problem_cfg)