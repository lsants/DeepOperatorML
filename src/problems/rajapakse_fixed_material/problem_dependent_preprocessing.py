from __future__ import annotations
import numpy as np
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def preprocess_raw_data(npz_filename: str, input_function_keys: list[str], coordinate_keys: list[str], **kwargs) -> dict[str, np.ndarray]:
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
      - Optionally, if the .npz file contains an operator output under the key 'g_u', it is also included.
    
    Args:
        npz_filename (str): Path to the .npz file.
        input_function_keys (list of str): List of keys for sensor (input function) arrays.
        coordinate_keys (list of str): List of keys for coordinate arrays.
    
    Returns:
        dict: A dictionary with the following keys:
            - 'xb': A 2D numpy array of shape (num_sensor_points, num_sensor_dimensions).
            - 'xt': A 2D numpy array of shape (num_coordinate_points, num_coordinate_dimensions).
            - 'g_u': (if present) the operator output array.
    """

    desired_direction = kwargs.get('direction')
    data = np.load(npz_filename, allow_pickle=True)
    
    input_funcs = [data[key] for key in input_function_keys]
    xb = np.column_stack(input_funcs)

    coords = [data[name] for name in coordinate_keys]
    coord_mesh = np.meshgrid(*coords, indexing='ij')
    xt = np.column_stack([m.flatten() for m in coord_mesh])
    
    result = {'xb': xb, 'xt': xt}
    if 'g_u' in data:
        result['g_u'] = data['g_u']
        if np.iscomplexobj(result['g_u']):
            result["g_u_real"] = result["g_u"].real
            result["g_u_imag"] = result["g_u"].imag
        if desired_direction:
            result['g_u'] = result['g_u'][..., desired_direction]
    else:
        raise ValueError("Operator target must be named 'g_u'")
    
    return result

def run_preprocessing(problem_settings: dict[str, any]):
    processed_data = preprocess_raw_data(problem_settings['RAW_DATA_PATH'],
                                         problem_settings['INPUT_FUNCTION_KEYS'],
                                         problem_settings['COORDINATE_KEYS']
                                         )

    save_path = Path(problem_settings['PROCESSED_DATA_PATH'])
    if save_path.parent:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **processed_data)
    logger.info(f"Succesfully saved processed data to {save_path}")

if __name__ == '__main__':
    problem = 'rajapakse_fixed_material'
    problem_path = "./configs/problems/" + 'rajapakse_fixed_material' + '/' + "config_problem.yaml"
    with open(problem_path) as f:
        problem_cfg = yaml.safe_load(f)
    run_preprocessing(problem_cfg)