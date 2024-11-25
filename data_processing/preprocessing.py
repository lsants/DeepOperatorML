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



def get_gaussian_norm_params(loader):
    samples_sum_xb = torch.zeros(1)
    samples_square_sum_xb = torch.zeros(1)
    
    samples_sum_g_u_real = torch.zeros(1)
    samples_square_sum_g_u_real = torch.zeros(1)

    samples_sum_g_u_imag = torch.zeros(1)
    samples_square_sum_g_u_imag = torch.zeros(1)

    for sample in loader:
        xb = sample['xb']
        g_u_real = sample['g_u_real']
        g_u_imag = sample['g_u_imag']

        samples_sum_xb += xb
        samples_sum_g_u_real += g_u_real
        samples_sum_g_u_imag += g_u_imag

        samples_square_sum_xb += xb.pow(2)
        samples_square_sum_g_u_real += g_u_real.pow(2)
        samples_square_sum_g_u_imag += g_u_imag.pow(2)

    samples_mean_xb = (samples_sum_xb / len(loader)).item()
    samples_std_xb = ((samples_square_sum_xb / len(loader) - samples_mean_xb.pow(2)).sqrt()).item()

    samples_mean_g_u_real = (samples_sum_g_u_real / len(loader)).item()
    samples_std_g_u_real = ((samples_square_sum_g_u_real / len(loader) - samples_mean_g_u_real.pow(2)).sqrt()).item()

    samples_mean_g_u_imag = (samples_sum_g_u_imag / len(loader)).item()
    samples_std_g_u_imag = ((samples_square_sum_g_u_imag / len(loader) - samples_mean_g_u_imag.pow(2)).sqrt()).item()

    gaussian_params = {'xb' : (samples_mean_xb, samples_std_xb),
                      'g_u_real' : (samples_std_g_u_real, samples_std_g_u_real),
                      'g_u_imag' :(samples_std_g_u_imag, samples_std_g_u_imag)}

    return gaussian_params

def trunk_to_meshgrid(arr):
    z = np.unique(arr[ : , 1])
    n_r = len(arr) / len(z)
    r = np.array(arr[ : , 0 ][ : int(n_r)]).flatten()
    return r, z

def meshgrid_to_trunk(r_values, z_values):
    R_mesh, Z_mesh = np.meshgrid(r_values, z_values)
    xt = np.column_stack((R_mesh.flatten(), Z_mesh.flatten()))
    return xt

def reshape_from_model(displacements, z_axis_values):
    if z_axis_values.ndim == 2:
        n_z = len(np.unique(z_axis_values[ : , 1]))
    else:
        n_z = len(z_axis_values)

    if isinstance(displacements, torch.Tensor):
        displacements = displacements.detach().numpy()

    if displacements.ndim == 3:
        displacements = (displacements).reshape(len(displacements), - 1, int(z_axis_values.shape[0] / n_z), n_z)

    if displacements.ndim == 2:
        displacements = (displacements).reshape(-1, int(z_axis_values.shape[0] / n_z), n_z)
    
    return displacements

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

def get_trunk_normalization_params(xt):
    r, z = trunk_to_meshgrid(xt)
    min_max_params = np.array([[r.min(), z.min()],
                                [r.max(), z.max()]])

    min_max_params = {'min' : [r.min(), z.min()],
                        'max' : [r.max(), z.max()]}
    return min_max_params


def get_pod_parameters(train_dataset, variance_share=0.95):
    """
    Computes the POD basis and mean functions from the training data.

    Args:
        train_dataset (torch.utils.data.Subset): Training dataset.
        num_modes (int): Number of POD modes to retain per output.

    Returns:
        torch.Tensor: POD basis matrices, shape (n_outputs, num_modes, features)
        torch.Tensor: Mean functions, shape (n_outputs, features)
    """
    n_outputs = train_dataset.dataset.n_outputs
    pod_basis_list = []
    mean_functions_list = []
    
    for i in range(n_outputs):
        if i == 0:
            outputs = torch.stack([train_dataset.dataset[idx]['g_u_real'] for idx in train_dataset.indices], dim=0)
            outputs = torch.stack([train_dataset.dataset[idx]['g_u_imag'] for idx in train_dataset.indices], dim=0)
        else:
            outputs = torch.stack([train_dataset.dataset[idx][f'g_u_{i}'] for idx in train_dataset.indices], dim=0)
        
        mean = torch.mean(outputs, dim=0)
        mean_functions_list.append(mean)
        
        centered = outputs - mean
        
        U, S, _ = torch.linalg.svd(centered)
        explained_variance_ratio = torch.cumsum(S**2, dim=0) / torch.linalg.norm(S)**2
        most_significant_modes = (explained_variance_ratio < variance_share).sum() + 1
        basis = U[ : , : most_significant_modes]
        
        pod_basis_list.append(basis)
    
    pod_basis = torch.stack(pod_basis_list, dim=0)
    mean_functions = torch.stack(mean_functions_list, dim=0)
    
    return pod_basis, mean_functions