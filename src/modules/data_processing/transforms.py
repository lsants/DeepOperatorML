from collections.abc import Callable, Iterable
from typing import Any
import torch
import numpy as np
from numpy import typing as npt

class Compose:
    def __init__(self, transforms: Iterable[Callable[..., Any]]) -> None:
        self.transforms = transforms

    def __call__(self, sample: Any) -> torch.TensorType | npt.ArrayLike:
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
class ToTensor:
    def __init__(self, dtype: torch.dtype, device: str) -> None:
        self.dtype = dtype
        self.device = torch.device(device=device)

    def __call__(self, sample: npt.ArrayLike) -> torch.Tensor:
        tensor = torch.tensor(data=sample, dtype=self.dtype, device=self.device)
        return tensor
    
class Rescale:
    def __init__(self, factor: float, config: str) -> None:
        """Rescaling DeepONet outputs (Nov. 2021, Fair comparison. Lu Lu, Karniadakis)

        Args:
            factor (float): Scaling factor (usually a function of the number of basis functions 'p')
            config (str): Function used for scaling. If 'none', scale by 1 (identity).
        """
        self.config = config
        self.factor = self.get_factor(factor=factor, config=self.config)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self.factor * sample
    
    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        return sample / self.factor
    
    def get_factor(self, factor: float, config: str) -> float:
        scales_config = {
            'none': 1,
            '1/p': 1 / factor,
            '1/sqrt(p)' : 1 / factor ** (0.5)
        }
        return scales_config[config]
    
    def update_scale_factor(self, new_scale_factor: float) -> None:
        self.factor = self.get_factor(new_scale_factor, self.config)

class NormalizeTransform:
    """Apply precomputed normalization parameters"""
    def __init__(self, scalers: dict[str, float], normalization_type: str):
        self.scalers = scalers
        self.normalization_type = normalization_type

    def __call__(self, sample):
        pass
class DenormalizeTransform:
    """Apply precomputed normalization parameters"""
    def __init__(self, scalers: dict[str, float], normalization_type: str):
        self.scalers = scalers
        self.normalization_type = normalization_type

    def __call__(self, sample):
        pass

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
