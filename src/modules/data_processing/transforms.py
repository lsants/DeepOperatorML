import numpy as np
import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
class ToTensor:
    def __init__(self, dtype: np.dtype, device: str) -> None:
        self.dtype = dtype
        self.device = device

    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        tensor = torch.tensor(sample, dtype=self.dtype, device=self.device)
        return tensor
    

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
