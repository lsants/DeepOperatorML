import torch
import numpy as np

class ToTensor:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def __call__(self, sample):
        tensor = torch.tensor(sample, dtype=self.dtype, device=self.device)
        return tensor
    
class Standardize:
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __call__(self, vals):
        vals = (vals - self.mu) / self.std
        return vals
    
class Destandardize:
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __call__(self, vals):
        vals = (vals * self.std) + self.mu
        return vals
    
class Normalize:
    def __init__(self, v_min, v_max):
        self.v_min = v_min
        self.v_max = v_max

    def __call__(self, vals):
        vals = (vals - self.v_min) / (self.v_max - self.v_min)
        return vals

class Denormalize:
    def __init__(self, v_min, v_max):
        self.v_min = v_min
        self.v_max = v_max

    def __call__(self, vals):
        vals = (vals * (self.v_max - self.v_min)) + self.v_min
        return vals

def get_branch_minmax_norm_params(loader):
    samples_min = torch.tensor([float('inf')])
    samples_max = torch.zeros(1)

    for sample in loader:
        xb = sample['xb']
        samples_min = min(xb.min(), samples_min)
        samples_max = max(xb.max(), samples_max)

    min_max_params = {'min':samples_min, 'max':samples_max}

    return min_max_params

def get_branch_gaussian_norm_params(loader):
    samples_sum = torch.zeros(1)
    samples_square_sum = torch.zeros(1)

    for sample in loader:
        xb = sample['xb']
        samples_sum += xb
        samples_square_sum += xb.pow(2)

    samples_mean = samples_sum / len(loader)
    samples_std = (samples_square_sum / len(loader) - samples_mean.pow(2)).sqrt()

    gaussian_params = {'mean':samples_mean, 'std':samples_std}

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

    return (displacements.detach().numpy()).reshape(len(displacements), -1, n_z)