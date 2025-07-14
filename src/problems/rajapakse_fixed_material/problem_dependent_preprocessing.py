from __future__ import annotations
import logging
import numpy as np
import numpy.typing as npt
from typing import Any, Iterable

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
                        processed_dataset_keys: dict[str, list[str]]) -> dict[str, npt.NDArray]:
    """
    Loads data from an npz file and groups the input functions and coordinates into tuples
    called 'xb' and 'xt' suitable for creating the PyTorch dataset.

    The function assumes that:
      - The input functions (sensors) are stored under keys given by input_function_keys.
        These may have different lengths. The function creates a meshgrid from these arrays
        (using 'ij' indexing) and then flattens the resulting arrays column‐wise to obtain a
        2D array of shape (num_sensor_points, num_sensor_dimensions).
      - The coordinate arrays (for the trunk) are stored under keys given by coordinate_keys.
        Again, a meshgrid is created and then flattened to yield a 2D array of shape
        (num_coordinate_points, num_coordinate_dimensions).

    Args:
        npz_filename (str): Path to the .npz file.
        input_function_keys (list of str): List of keys for sensor (input function) arrays.
        coordinate_keys (list of str): List of keys for coordinate arrays.

    Returns:
        dict: A dictionary with the following keys:
            - 'xb': A 2D numpy array of shape (num_sensor_points, num_sensor_dimensions).
            - 'xt': A 2D numpy array of shape (num_coordinate_points, num_coordinate_dimensions).
            - 'g_u': the operator output array.
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
    processed_data = features

    for i in processed_dataset_keys['targets']:
        if i in data:
            placeholder = []
            if np.iscomplexobj(data[i]):
                g_u_real = data[i].real
                g_u_imag = data[i].imag
                placeholder.append(g_u_real)
                placeholder.append(g_u_imag)
            processed_data[i] = np.stack(placeholder, axis=3)
            processed_data[i] = processed_data[i].reshape(len(branch_input),
                                                          len(trunk_input),
                                                          -1)
        else:
            raise KeyError(
                f"Operator target '{processed_dataset_keys['targets'][0]}' must be present in the dataset keys")
    return processed_data


def run_preprocessing(problem_settings: dict[str, Any]) -> dict[str, npt.NDArray]:
    processed_data = preprocess_raw_data(raw_npz_filename=problem_settings['raw_data_path'],
                                         input_function_keys=problem_settings['input_function_keys'],
                                         coordinate_keys=problem_settings['coordinate_keys'],
                                         processed_dataset_keys=problem_settings['data_labels']
                                         )
    return processed_data
