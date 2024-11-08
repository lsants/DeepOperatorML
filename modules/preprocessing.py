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
        v_min = self.v_min
        v_max = self.v_max
        if not isinstance(self.v_min, torch.Tensor) or not isinstance(self.v_max, torch.Tensor):
            v_min = torch.tensor(self.v_min)
            v_max = torch.tensor(self.v_max)

        vals = (vals - v_min) / (v_max - v_min)
        return vals

class Denormalize:
    def __init__(self, v_min, v_max):
        self.v_min = v_min
        self.v_max = v_max

    def __call__(self, vals):
        v_min = self.v_min
        v_max = self.v_max
        if not isinstance(self.v_min, torch.Tensor) or not isinstance(self.v_max, torch.Tensor):
            v_min = torch.tensor(self.v_min)
            v_max = torch.tensor(self.v_max)
        vals = (vals * (v_max - v_min)) + v_min
        return vals

def get_minmax_norm_params(loader):
    samples_min_xb = torch.tensor([float('inf')])
    samples_max_xb = torch.zeros(1)

    samples_min_g_u_real = torch.tensor([float('inf')])
    samples_max_g_u_real = torch.zeros(1)

    samples_min_g_u_imag = torch.tensor([float('inf')])
    samples_max_g_u_imag = torch.zeros(1)

    for sample in loader:
        xb = sample['xb']
        g_u_real = sample['g_u_real']
        g_u_imag = sample['g_u_imag']
        samples_min_xb = min(xb.min(), samples_min_xb)
        samples_max_xb = max(xb.max(), samples_max_xb)

        samples_min_g_u_real = min(g_u_real.min(), samples_min_g_u_real)
        samples_max_g_u_real = max(g_u_real.max(), samples_max_g_u_real)

        samples_min_g_u_imag = min(g_u_imag.min(), samples_min_g_u_imag)
        samples_max_g_u_imag = max(g_u_imag.max(), samples_max_g_u_imag)

    if isinstance(samples_min_xb, torch.Tensor) or isinstance(samples_min_g_u_real, torch.Tensor) or isinstance(samples_min_g_u_imag, torch.Tensor):
        samples_min_xb = samples_min_xb.item()
        samples_max_xb = samples_max_xb.item()
    
        samples_min_g_u_real = samples_min_g_u_real.item()
        samples_max_g_u_real = samples_max_g_u_real.item()
    
        samples_min_g_u_imag = samples_min_g_u_imag.item()
        samples_max_g_u_imag = samples_max_g_u_imag.item()

    min_max_params = {'xb' : {'min' : samples_min_xb, 'max' : samples_max_xb},
                      'g_u_real' : {'min' : samples_min_g_u_real, 'max' : samples_max_g_u_real},
                      'g_u_imag' : {'min' : samples_min_g_u_imag, 'max' : samples_max_g_u_imag}}

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
    
    return (displacements).reshape(len(displacements), int(displacements.shape[-1] / n_z), n_z)

def trunk_feature_expansion(xt, p):
    expansion_features = [xt]
    if p:
        for k in range(1, p + 1):
            expansion_features.append(torch.sin(k * torch.pi * xt))
            expansion_features.append(torch.cos(k * torch.pi * xt))

    trunk_features = torch.concat(expansion_features, axis=1)
    return trunk_features