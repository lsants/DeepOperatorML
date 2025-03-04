import logging
import torch
import numpy as np
from ..deeponet.deeponet import DeepONet
from ..data_processing.deeponet_dataset import DeepONetDataset

logger = logging.getLogger(__name__)
class ToTensor:
    def __init__(self, dtype: np.dtype, device: str) -> None:
        self.dtype = dtype
        self.device = device

    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        tensor = torch.tensor(sample, dtype=self.dtype, device=self.device)
        return tensor
    
class Scaling:
    def __init__(self, min_val: float | torch.Tensor | None=None, max_val: float | torch.Tensor | None=None, mean: float | torch.Tensor | None=None, std: float | torch.Tensor | None=None) -> None:
        """
        A generic class for scaling values, supporting both normalization and standardization.

        Args:
            min_val (float or Tensor, optional): Minimum value for normalization.
            max_val (float or Tensor, optional): Maximum value for normalization.
            mean (float or Tensor, optional): Mean value for standardization.
            std (float or Tensor, optional): Standard deviation for standardization.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.mean = mean
        self.std = std

        if not ((min_val is not None and max_val is not None) or (mean is not None and std is not None)):
            raise ValueError("Either min_val and max_val or mean and std must be provided.")

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Normalizes values to the range [0, 1].

        Args:
            values (Tensor): Input values.

        Returns:
            Tensor: Normalized values.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("min_val and max_val must be provided for normalization.")
        
        v_min = torch.as_tensor(self.min_val, dtype=values.dtype, device=values.device)
        v_max = torch.as_tensor(self.max_val, dtype=values.dtype, device=values.device)
        return (values - v_min) / (v_max - v_min)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Denormalizes values from the range [0, 1] back to the original range.

        Args:
            values (Tensor): Input normalized values.

        Returns:
            Tensor: Denormalized values.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("min_val and max_val must be provided for denormalization.")
        
        v_min = torch.as_tensor(self.min_val, dtype=values.dtype, device=values.device)
        v_max = torch.as_tensor(self.max_val, dtype=values.dtype, device=values.device)
        return values * (v_max - v_min) + v_min

    def standardize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Standardizes values using the provided mean and standard deviation.

        Args:
            values (Tensor): Input values.

        Returns:
            Tensor: Standardized values.
        """
        if self.mean is None or self.std is None:
            raise ValueError("mean and std must be provided for standardization.")
        
        mu = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        sigma = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return (values - mu) / sigma

    def destandardize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Destandardizes values back to the original scale using mean and standard deviation.

        Args:
            values (Tensor): Input standardized values.

        Returns:
            Tensor: Destandardized values.
        """
        if self.mean is None or self.std is None:
            raise ValueError("mean and std must be provided for destandardization.")
        
        mu = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        sigma = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return values * sigma + mu

def preprocess_npz_data(npz_filename: str, input_function_keys: list[str], coordinate_keys: list[str], **kwargs) -> dict[str, torch.Tensor]:
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
    sensor_mesh = np.meshgrid(*input_funcs, indexing='ij')
    xb = np.column_stack([m.flatten() for m in sensor_mesh])

    if xb.ndim == 1:
        xb = xb.reshape(len(xb), -1)
    
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

    normalize_branch = Scaling(min_val=xb_min, max_val=xb_max)
    normalize_trunk = Scaling(min_val=xt_min, max_val=xt_max)

    normalization_parameters = {
        "xb": {
            "min": xb_min,
            "max": xb_max,
            "normalize": normalize_branch.normalize,
            "denormalize": normalize_branch.denormalize
        },
        "xt": {
            "min": xt_min,
            "max": xt_max,
            "normalize": normalize_trunk.normalize,
            "denormalize": normalize_trunk.denormalize
        }
    }

    for key in params['OUTPUT_KEYS']:
        key_min, key_max = min_max_vals[key]['min'], min_max_vals[key]['max']
        scaling = Scaling(min_val=key_min, max_val=key_max)
        normalization_parameters[key] = {
            "min": key_min,
            "max": key_max,
            "normalize": scaling.normalize,
            "denormalize": scaling.denormalize
        }
    return normalization_parameters


def prepare_batch(batch: dict[str, torch.Tensor], params: dict[str, any]) -> dict[str, torch.Tensor]:
    """
    Prepares the batch data, including normalization and feature expansion.

    Args:
        batch (dict): The batch data.

    Returns:
        dict: The processed batch data.
    """
    processed_batch = {}
    dtype = getattr(torch, params['PRECISION'])
    device = params['DEVICE']

    xb_scaler = Scaling(
        min_val = params['NORMALIZATION_PARAMETERS']['xb']['min'],
        max_val = params['NORMALIZATION_PARAMETERS']['xb']['max']
    )
    xt_scaler = Scaling(
        min_val = params['NORMALIZATION_PARAMETERS']['xt']['min'],
        max_val = params['NORMALIZATION_PARAMETERS']['xt']['max']
    )

    if params['INPUT_NORMALIZATION']:
        processed_batch['xb'] = xb_scaler.normalize(batch['xb']).to(dtype=dtype, device=device)
        processed_batch['xt'] = xt_scaler.normalize(batch['xt']).to(dtype=dtype, device=device)
    else:
        processed_batch['xb'] = batch['xb'].to(dtype=dtype, device=device)
        processed_batch['xt'] = batch['xt'].to(dtype=dtype, device=device)

    for key in params['OUTPUT_KEYS']:
        scaler = Scaling(
            min_val = params['NORMALIZATION_PARAMETERS'][key]['min'],
            max_val = params['NORMALIZATION_PARAMETERS'][key]['max']
        )
        if params['OUTPUT_NORMALIZATION']:
            processed_batch[key] = scaler.normalize(batch[key]).to(dtype=dtype, device=device)
        else:
            processed_batch[key] = batch[key].to(dtype=dtype, device=device)

    if params['TRUNK_FEATURE_EXPANSION']:
        processed_batch['xt'] = trunk_feature_expansion(
            processed_batch['xt'], params['TRUNK_EXPANSION_FEATURES_NUMBER']
        )

    return processed_batch

def get_single_batch(dataset: DeepONetDataset, indices, params) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, params['PRECISION'])
    device = params['DEVICE']

    batch = {}
    batch['xb'] = torch.stack([dataset[idx]['xb'] for idx in indices], dim=0).to(dtype=dtype, device=device)
    batch['xt'] = dataset.get_trunk()
    for key in params['OUTPUT_KEYS']:
        batch[key] = torch.stack([dataset[idx][key] for idx in indices], dim=0).to(dtype=dtype, device=device)
    return batch

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

def process_outputs_to_plot_format(output: torch.Tensor, coords, basis: bool=False) -> np.ndarray:
    """
    Reshapes the output from DeepONet into a meshgrid format for plotting.
    
    Args:
        output (Tensor or ndarray): The network output, with shape either
            (branch_data.shape[0], trunk_size) or 
            (branch_data.shape[0], trunk_size, n_basis).
        coords (tuple or list or ndarray): The coordinate arrays that were
            used to generate the trunk. If multiple coordinate arrays are provided,
            they should be in a tuple/list (e.g. (x_values, y_values, z_values)).
            If a single array is provided, it is assumed to be 1D.
    
    Returns:
        ndarray: Reshaped output with shape 
            (branch_data.shape[0], n_basis, len(coords[0]), len(coords[1]), ...).
            For a 2D problem with a single basis, for example, the output shape
            will be (N_branch, 1, len(coord1), len(coord2)).
    """
    if isinstance(coords, (list, tuple)):
        grid_shape = tuple(len(c) for c in coords)
    else:
        grid_shape = (len(coords),)
    
    output = output.detach().cpu().numpy()

    if output.ndim == 2:
        N_branch, trunk_size = output.shape
        if np.prod(grid_shape) != trunk_size and not basis:
            raise ValueError("Mismatch between trunk size and product of coordinate lengths.")
        reshaped = output.reshape(N_branch, *grid_shape, 1)
    elif output.ndim == 3:
        if not basis:
            N_branch, outputs, trunk_size = output.shape
        else:
            N_branch, trunk_size, outputs = output.shape

        if np.prod(grid_shape) != trunk_size:
                raise ValueError("Mismatch between trunk size and product of coordinate lengths.")
        reshaped = output.reshape(N_branch, *grid_shape, outputs)

    else:
        raise ValueError("Output must be either 2D or 3D.")
    
    return reshaped

def trunk_feature_expansion(xt: torch.Tensor, n_exp_features: int) -> torch.Tensor:
    expansion_features = [xt]
    if n_exp_features:
        for k in range(1, n_exp_features + 1):
            expansion_features.append(torch.sin(k * torch.pi * xt))
            expansion_features.append(torch.cos(k * torch.pi * xt))

    trunk_features = torch.concat(expansion_features, axis=1)
    return trunk_features

def mirror(arr: np.ndarray) -> np.ndarray:
    arr_flip = np.flip(arr[1 : , : ], axis=1)
    arr_mirrored = np.concatenate((arr_flip, arr), axis=1)
    arr_mirrored = arr_mirrored.T
    return arr_mirrored

def format_param(param: dict[str, any], param_keys: list[str] | tuple | None=None) -> str:
    """
    Format a parameter value for display in the plot title.
    
    - If 'param' is a dict, it returns a string of the form:
          (key1=value1, key2=value2, ...)
    - If 'param' is an iterable (but not a string) and param_keys is provided and its length
      matches the length of 'param', it returns a string of the form:
          (key1=value1, key2=value2, ...)
      where the keys are taken from param_keys.
    - Otherwise, it returns the string representation of param.
    
    Args:
        param: The parameter value, which can be a dict, tuple, list, etc.
        param_keys (list or tuple, optional): List of keys to use if 'param' is an iterable.
    
    Returns:
        str: The formatted parameter string.
    """
    if isinstance(param, dict):
        items = [f"{k}={f'{v:.2f}'}" for k, v in param.items()]
        return "(" + ", ".join(items) + ")"
    elif hasattr(param, '__iter__') and not isinstance(param, str):
        if param_keys is not None and len(param_keys) == len(param):
            items = [f"{k}={f'{v:.2f}'}" for k, v in zip(param_keys, param)]
            return "(" + ", ".join(items) + ")"
        else:
            items = [f"{v:.2f}" for v in param]
            return "(" + ", ".join(items) + ")"
    else:
        return str(param)
    
def postprocess_for_2D_plot(model: DeepONet, plot_config: dict[str, any], model_config: dict[str, any], 
                            branch_features: torch.Tensor, trunk_features: torch.Tensor,
                            ground_truth: torch.Tensor, preds: torch.Tensor) -> dict[str, np.ndarray]:
    processed_data = {}

    # -------------- Prepare branch data --------------

    xb_keys = model_config["INPUT_FUNCTION_KEYS"]

    xb_scaler = Scaling(
        min_val=model_config['NORMALIZATION_PARAMETERS']['xb']['min'],
        max_val=model_config['NORMALIZATION_PARAMETERS']['xb']['max']
    )
    if model_config['INPUT_NORMALIZATION']:
        branch_features = xb_scaler.denormalize(branch_features)
    branch_tuple = don_to_meshgrid(branch_features)
    branch_map = {k:v for k, v in zip(xb_keys, branch_tuple)}
    processed_data["branch_features"] = np.array(branch_features)
    processed_data["branch_map"] = branch_map

    # -------------- Prepare trunk data --------------

    xt_scaler = Scaling(
        min_val=model_config['NORMALIZATION_PARAMETERS']['xt']['min'],
        max_val=model_config['NORMALIZATION_PARAMETERS']['xt']['max']
    )
    xt_plot = trunk_features
    if model_config['TRUNK_FEATURE_EXPANSION']:
        xt_plot = xt_plot[:, : xt_plot.shape[-1] // (1 + 2 * model_config['TRUNK_EXPANSION_FEATURES_NUMBER'])]
    if model_config['INPUT_NORMALIZATION']:
        xt_plot = xt_scaler.denormalize(xt_plot)

    if "COORDINATE_KEYS" not in model_config:
        raise ValueError("COORDINATE_KEYS must be provided in the configuration.")
    coordinate_keys = model_config["COORDINATE_KEYS"]  # e.g., ["x", "y", "z"]
    coords_tuple = don_to_meshgrid(xt_plot)
    if len(coords_tuple) != len(coordinate_keys):
        raise ValueError("Mismatch between number of coordinates in trunk data and COORDINATE_KEYS.")
    
    coordinates_map = {k: v for k, v in zip(coordinate_keys, coords_tuple)}
    coord_index_map = {coord: index for index, coord in enumerate(coordinates_map)}
    coords_2D_index_map = {k: v for k, v in coord_index_map.items() if k in plot_config["AXES_TO_PLOT"]}

    if len(coord_index_map) > 2:
        index_to_remove_coords = [coord_index_map[coord] for coord in coord_index_map if coord not in coords_2D_index_map][0]
    else:
        index_to_remove_coords = None
    col_indices = [index for index in coord_index_map.values() if index != index_to_remove_coords]
    coords_2D_map = {k : v for k, v in coordinates_map.items() if k in plot_config["AXES_TO_PLOT"]}

    processed_data["coords_2D"] = coords_2D_map
    processed_data["trunk_features"] = xt_plot


    if len(coord_index_map) > 2:
        processed_data["trunk_features_2D"] = processed_data["trunk_features"][ : , col_indices]
    else:
        processed_data["trunk_features_2D"] = processed_data["trunk_features"]
    
    # ------------------ Prepare outputs ---------------------

    output_keys = model_config["OUTPUT_KEYS"]
    if len(output_keys) == 2:
        truth_field = ground_truth[output_keys[0]] + ground_truth[output_keys[1]] * 1j
        pred_field = preds[output_keys[0]] + preds[output_keys[1]] * 1j
    else:
        truth_field = ground_truth[output_keys[0]]
        pred_field = preds[output_keys[0]]

    truth_field = process_outputs_to_plot_format(truth_field, coords_tuple)
    pred_field = process_outputs_to_plot_format(pred_field, coords_tuple)

    trunk_output = model.training_strategy.get_basis_functions(xt=trunk_features, model=model)
    # branch_output = model.training_strategy.get_coefficients(xb=branch_features, model=model)
    basis_modes = process_outputs_to_plot_format(trunk_output, coords_tuple, basis=True)
    # coeff_modes = process_outputs_to_plot_format(branch_output, branch_tuple, basis=True) # Need to implement a 'plot coeffs' function for the future
    
    if basis_modes.ndim < 4:
        basis_modes = np.expand_dims(basis_modes, axis=1)
    
    if basis_modes.shape[0] > model_config.get('BASIS_FUNCTIONS'):
        split_1 = basis_modes[ : model_config.get('BASIS_FUNCTIONS')]
        split_2 = basis_modes[model_config.get('BASIS_FUNCTIONS') : ]
        basis_modes = np.concatenate([split_1, split_2], axis=-1)

    truth_slicer = [slice(None)] * truth_field.ndim
    pred_slicer = [slice(None)] * pred_field.ndim
    basis_slicer = [slice(None)] * basis_modes.ndim
    if index_to_remove_coords:
        processed_data["index_to_remove_coords"] = index_to_remove_coords
        truth_slicer[index_to_remove_coords + 2] = 0
        pred_slicer[index_to_remove_coords + 2] = 0
        basis_slicer[index_to_remove_coords + 1] = 0
    
    basis_modes_sliced = basis_modes[tuple(basis_slicer)]
    
    processed_data["output_keys"] = output_keys
    processed_data["truth_field"] = truth_field
    processed_data["pred_field"] = pred_field
    processed_data["truth_field_2D"] = truth_field[tuple(truth_slicer)]
    processed_data["pred_field_2D"] = pred_field[tuple(pred_slicer)]
    processed_data["basis_functions_2D"] = basis_modes_sliced
    
    logger.info(f"\nOutputs shape: {pred_field.shape}\n")
    logger.info(f"\n2D Outputs shape: {processed_data['pred_field_2D'].shape}\n")
    logger.info(f"\n2D Truths shape: {processed_data['truth_field_2D'].shape}\n")
    logger.info(f"\n2D Basis functions shape: {processed_data['basis_functions_2D'].shape}\n")

    return processed_data