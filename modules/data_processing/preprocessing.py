import torch
import numpy as np

class ToTensor:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def __call__(self, sample):
        tensor = torch.tensor(sample, dtype=self.dtype, device=self.device)
        return tensor
    
class Scaling:
    def __init__(self, min_val=None, max_val=None, mean=None, std=None):
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

    def normalize(self, values):
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

    def denormalize(self, values):
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

    def standardize(self, values):
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

    def destandardize(self, values):
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
 
def prepare_data_for_dataset(data): # FINISH THIS FUNCTION SO THAT DIMENSIONS ARE CORRECT FOR DATASET
    xb = data['']


def get_minmax_norm_params(dataset, keys=None):
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

def don_to_meshgrid(arr):
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

def meshgrid_to_don(*coords):
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


def reshape_outputs_to_plot_format(output, coords):
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
    
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    
    if output.ndim == 2:
        N_branch, trunk_size = output.shape
        if np.prod(grid_shape) != trunk_size:
            raise ValueError("Mismatch between trunk size and product of coordinate lengths.")
        reshaped = output.reshape(N_branch, 1, *grid_shape)
    elif output.ndim == 3:
        N_branch, trunk_size, n_basis = output.shape
        if np.prod(grid_shape) != trunk_size:
            raise ValueError("Mismatch between trunk size and product of coordinate lengths.")
        reshaped = output.reshape(N_branch, n_basis, *grid_shape)
    else:
        raise ValueError("Output must be either 2D or 3D.")
    
    return reshaped

def trunk_feature_expansion(xt, p):
    expansion_features = [xt]
    if p:
        for k in range(1, p + 1):
            expansion_features.append(torch.sin(k * torch.pi * xt))
            expansion_features.append(torch.cos(k * torch.pi * xt))

    trunk_features = torch.concat(expansion_features, axis=1)
    return trunk_features

def mirror(arr):
    arr_flip = np.flip(arr[1 : , : ], axis=1)
    arr_mirrored = np.concatenate((arr_flip, arr), axis=1)
    arr_mirrored = arr_mirrored.T
    return arr_mirrored