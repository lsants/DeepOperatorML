import torch
import numpy as np
from .scaling import Scaling
from .deeponet_dataset import DeepONetDataset
from .process_branch_input import branch_processing_map

def preprocess_npz_data(npz_filename: str, input_function_keys: list[str], coordinate_keys: list[str], branch_processing_strategy: str, **kwargs) -> dict[str, torch.Tensor]:
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
    xb = branch_processing_map[branch_processing_strategy]().compute_xb(input_funcs)

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

def get_minmax_norm_params(dataset: DeepONetDataset, keys: list[str] | None=None) ->  dict[str, dict[str, float]]:
    """
    Compute min-max normalization parameters for specified keys in the dataset.

    Args:
        dataset (torch.utils.data.Dataset or torch.utils.data.Subset): Dataset or subset.
        keys (list of str, optional): Keys to normalize. If None, includes 'xb', 'xt', and all outputs.

    Returns:
        dict: Dictionary containing min and max values for each key.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices
    else:
        original_dataset = dataset
        indices = range(len(dataset))

    if keys is None:
        keys = ['xb', 'xt'] + getattr(original_dataset, 'output_keys', [])

    min_max_params = {key: {'min': float('inf'), 'max': -float('inf')} for key in keys}

    for idx in indices:
        sample = original_dataset[idx]

        for key in keys:
            if key == 'xt':
                values = original_dataset.get_trunk()
            else:
                values = sample[key]

            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()

            min_max_params[key]['min'] = min(min_max_params[key]['min'], np.min(values))
            min_max_params[key]['max'] = max(min_max_params[key]['max'], np.max(values))

    return min_max_params

def get_norm_params(train_dataset: dict[str, torch.utils.data.Subset], params: dict[str, any]) -> dict[str, any]:
    min_max_vals = get_minmax_norm_params(train_dataset)

    xb_min, xb_max = min_max_vals['xb']['min'], min_max_vals['xb']['max']
    xt_min, xt_max = min_max_vals['xt']['min'], min_max_vals['xt']['max']

    normalization_parameters = {
        "xb": {
            "min": xb_min,
            "max": xb_max,
        },
        "xt": {
            "min": xt_min,
            "max": xt_max,
        }
    }   
    for key in params['OUTPUT_KEYS']:
        key_min, key_max = min_max_vals[key]['min'], min_max_vals[key]['max']
        normalization_parameters[key] = {
            "min": key_min,
            "max": key_max,
        }
    return normalization_parameters

def don_to_meshgrid(arr: np.ndarray) -> tuple[np.ndarray]:
    """
    Recovers the original coordinate arrays from a trunk (or branch) array.
    
    Assumes that the trunk array was created via a meshgrid operation (with 'ij' indexing)
    from one or more 1D coordinate arrays. The trunk is a 2D array of shape (N, d) where d is
    the number of coordinate dimensions and N is the product of the lengths of the coordinate arrays.
    
    Returns a tuple of d 1D arrays containing the unique coordinate values for each dimension.
    For example, for a 2D case it returns (r, z); for a 3D case, (x, y, z).
    
    Args:
        arr (numpy.ndarray): Trunk array of shape (N, d).
    
    Returns:
        tuple: A tuple of d 1D numpy arrays corresponding to the coordinates.
    """
    d = arr.shape[1]
    coords = tuple(np.unique(arr[:, i]) for i in range(d))
    return coords

def meshgrid_to_don(*coords: np.ndarray) -> np.ndarray:
    """
    Generates the trunk/branch matrix for DeepONet training from given coordinate arrays.
    
    This function accepts either multiple coordinate arrays as separate arguments,
    or a single argument that is a list (or tuple) of coordinate arrays. It returns
    a 2D array where each row corresponds to one point in the Cartesian product of the
    input coordinate arrays.
    
    Examples:
        For 2D coordinates:
            xt = meshgrid_to_trunk(r_values, z_values)
        For 3D coordinates:
            xt = meshgrid_to_trunk(x_values, y_values, z_values)
        Or:
            xt = meshgrid_to_trunk([x_values, y_values, z_values])
    
    Args:
        *coords: One or more 1D numpy arrays representing coordinate values.
    
    Returns:
        numpy.ndarray: A 2D array of shape (N, d) where d is the number of coordinate arrays
                       and N is the product of the lengths of these arrays.
    """

    if len(coords) == 1 and isinstance(coords[0], (list, tuple)):
        coords = coords[0]
    
    meshes = np.meshgrid(*coords, indexing='ij')

    data = np.column_stack([m.flatten() for m in meshes])
    return data