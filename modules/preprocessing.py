import torch
import numpy as np

class ToTensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, sample):
        tensor = torch.tensor(sample, dtype=self.dtype)
        return tensor
    
class Normalize:
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __call__(self, tensor):
        tensor = (tensor - self.mu) / self.std

class Denormalize:
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __call__(self, tensor):
        tensor = (tensor * self.std) + self.mu

def get_branch_normalization_params(loader):
    samples_sum = torch.zeros(1)
    samples_square_sum = torch.zeros(1)
    samples_min = torch.tensor([float('inf')])
    samples_max = torch.zeros(1)

    for sample in loader:
        xb = sample['xb']
        samples_sum += xb
        samples_square_sum += xb.pow(2)
        samples_min = min(xb, samples_min)
        samples_max = max(xb, samples_max)

    samples_mean = samples_sum / len(loader)
    samples_std = (samples_square_sum / len(loader) - samples_mean.pow(2)).sqrt()

    gaussian_params = {'mean':samples_mean.tolist(), 'std':samples_std.tolist()}
    min_max_params = {'min':samples_min.tolist(), 'max':samples_max.tolist()}

    return gaussian_params, min_max_params

def trunk_to_meshgrid(arr):
    z = np.unique(arr[ : , 1])
    n_r = len(arr) / len(z)
    r = arr[ : , 0 ][ : int(n_r)]
    return r, z

def meshgrid_to_trunk(r_values, z_values):
    R_mesh, Z_mesh = np.meshgrid(r_values, z_values)
    xt = np.column_stack((R_mesh.flatten(), Z_mesh.flatten()))
    return xt