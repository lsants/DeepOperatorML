from __future__ import annotations
import torch
import numpy as np
import pickle
from . import transforms
from collections.abc import Callable, Iterable
from typing import Any, Dict

class   DeepONetTransformer:
    """Handles custom transforms for DeepONet with serialization support."""

    def __init__(self, config: Dict[str, Any]):
        # Configuration for transformations
        self.config = config

        # Initialize transformation components
        self.branch_scaler = None
        self.trunk_scaler = None
        self.output_rescalers = {}
        self.n_exp_features = config.get('n_exp_features', 0)
        self.device = config.get('device', 'cpu')
        self.dtype = config.get('dtype', torch.float32)

        self.branch_transforms = transforms.Compose(transforms=[
            transforms.ToTensor(dtype=self.dtype, device=self.device),
            self._create_branch_normalizer()
        ])

        self.trunk_transforms = transforms.Compose(transforms=[
            self._create_trunk_feature_expansion(),
            transforms.ToTensor(dtype=self.dtype, device=self.device)
        ])

    def _create_branch_normalizer(self) -> Callable:
        """Create normalization pipeline for branch inputs"""
        return transforms.Normalize(scalers=self.branch_scaler,
                         normalization_type=self.config.get('branch_norm', 'zscore'))

    def _create_trunk_feature_expansion(self) -> Callable:
        """Create feature expansion for trunk inputs"""
        def _expander(x: np.ndarray) -> torch.Tensor:
            xt_tensor = torch.as_tensor(x, dtype=self.dtype)
            return transforms.trunk_feature_expansion(xt=xt_tensor, n_exp_features=self.n_exp_features)
        return _expander

    def fit(self, data: Dict[str, np.ndarray]):
        """Calculate normalization parameters"""
        # Branch input normalization
        if 'branch_norm' in self.config:
            self.branch_scaler = {
                'mean': data['xb'].mean(axis=0),
                'std': data['xb'].std(axis=0)
            }

        # Initialize output rescalers
        for key in data['output_keys']:
            self.output_rescalers[key] = transforms.Rescale(
                factor=self.config.get('scale_factor', 1),
                config=self.config.get('rescaling', '1/sqrt(p)')
            )

    def transform_branch(self, xb: np.ndarray, training: bool = False) -> torch.Tensor:
        """Apply branch transforms with optional training-time noise"""
        transformed = self.branch_transforms(xb)
        if training and self.config.get('add_noise', False):
            transformed += torch.randn_like(transformed) * \
                self.config.get('noise_std', 0.1)
        return transformed

    def transform_trunk(self, xt: np.ndarray) -> torch.Tensor:
        """Apply trunk feature expansion"""
        return self.trunk_transforms(xt)

    def transform_output(self, y: np.ndarray, key: str) -> torch.Tensor:
        """Apply output rescaling"""
        return self.output_rescalers[key](torch.as_tensor(y, dtype=self.dtype))

    def inverse_transform_output(self, y: torch.Tensor, key: str) -> torch.Tensor:
        """Inverse output rescaling"""
        return self.output_rescalers[key].inverse(y)

    def save(self, path: str):
        """Serialize transformer state"""
        state = {
            'config': self.config,
            'branch_scaler': self.branch_scaler,
            'output_rescalers': self.output_rescalers,
            'n_exp_features': self.n_exp_features
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        """Load serialized transformer"""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        if device is not None:
            state['config']['device'] = device

        transformer = cls(state['config'])
        transformer.branch_scaler = state['branch_scaler']
        transformer.output_rescalers = state['output_rescalers']
        transformer.n_exp_features = state['n_exp_features']
        return transformer