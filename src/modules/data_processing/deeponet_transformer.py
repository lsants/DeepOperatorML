from __future__ import annotations
from re import S
import torch
import numpy as np
from collections.abc import Callable
from typing import Optional, Literal
from .config import TransformConfig
from pathlib import Path
from .data_augmentation.feature_expansions import FeatureExpansionConfig
from .data_augmentation.feature_expansions import FeatureExpansionRegistry
from .config import ComponentTransformConfig

class DeepONetTransformPipeline:
    def __init__(self, config: TransformConfig):
        self.config = config
        self.branch_stats: dict = {}  # Track mean/std for branch
        self.trunk_stats: dict = {}  # Track mean/std for trunk
        self.original_dims: dict = {}
        self.expansion_factors: dict = {}

        # Initialize branch components
        self.branch_normalizer = self._init_normalizer(
            norm_type=config.branch.normalization, component='branch' # type: ignore
        )
        self.branch_expander = self._init_expander(
            expansion_cfg=config.branch.feature_expansion, component='branch' # type: ignore
        )

        # Initialize trunk components
        self.trunk_normalizer = self._init_normalizer(
            norm_type=config.trunk.normalization, component='trunk' # type: ignore
        )
        self.trunk_expander = self._init_expander(
            expansion_cfg=config.trunk.feature_expansion, component='trunk' # type: ignore
        )

    def _init_normalizer(self, norm_type: str, component: str) -> Optional[Callable]:
        """Returns normalization function and stores stats"""
        if not norm_type: return None
        
        def normalizer(data: torch.Tensor) -> torch.Tensor:
            if norm_type == "standardize":
                mean = data.mean(dim=0)
                std = data.std(dim=0)
                self._update_stats([f'{component}_norm'] , {'mean': mean, 'std': std})
                return (data - mean) / std
            if norm_type == "0_1_min_max":
                min_val = torch.min(data, dim=0)
                max_val = torch.max(data, dim=0)
                self._update_stats([f'{component}_norm'] , {'max': max_val, 'min': min_val})
                return (data - min_val) / (max_val - min_val) # type: ignore
            if norm_type == "-1_1_min_max":
                min_val = torch.min(data, dim=0)
                max_val = torch.max(data, dim=0)
                self._update_stats([f'{component}_norm'] , {'max': max_val, 'min': min_val})
                return 2*(data - min_val) / (max_val - min_val) - 1 # type: ignore
            # Implement other norm types similarly
            return data
        
        return normalizer

    def _init_expander(self, expansion_cfg: FeatureExpansionConfig, component: str) -> Optional[Callable]:
        """Retrieve expansion from registry"""
        if not expansion_cfg: return None
        self.expansion_factors[component] = {
            'original_dim': expansion_cfg.original_dim,  # Set during first transform
            'expansion_type': expansion_cfg.type,
            'size': expansion_cfg.size
        }

        return self._get_expansion_fn(expansion_type=expansion_cfg.type, component=component)
    
    def _get_expansion_fn(self, expansion_type: str, component: str) -> Callable:
        """Registry wrapper with dimension tracking"""
        def expander(x: torch.Tensor) -> torch.Tensor:
            # Record original dimension on first use
            if self.expansion_factors[component]['original_dim'] is None:  # Changed check
                self.expansion_factors[component]['original_dim'] = x.shape[-1]
            
            # Get expansion function and call it with x
            expansion_fn = FeatureExpansionRegistry.get_expansion_fn(
                name=expansion_type,
                size=self.expansion_factors[component]['size']
            )
            return expansion_fn(x)  # Now passing x to the expansion function
            
        return expander


    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, dtype=getattr(torch, self.config.dtype)).to(self.config.device)

    def transform_branch(self, xb: np.ndarray) -> torch.Tensor:
        tensor = self._to_tensor(xb)
        if self.branch_normalizer:
            tensor = self.branch_normalizer(tensor)
        return tensor

    def transform_trunk(self, xt: np.ndarray) -> torch.Tensor:
        tensor = self._to_tensor(xt)
        if self.trunk_normalizer:
            tensor = self.trunk_normalizer(tensor)
        if self.trunk_expander:
            tensor = self.trunk_expander(tensor)
        return tensor

    def save(self, path: Path):
        """Serializes exactly what your architecture needs for inference"""
        state = {
            'branch_stats': self.branch_stats,
            'trunk_stats': self.trunk_stats,
            'config': {
                'device': self.config.device,
                'dtype': str(self.config.dtype).split('.')[-1],
                'branch_feature_expansion': self.config.branch.feature_expansion,
                'trunk_feature_expansion': self.config.trunk.feature_expansion,
                'normalization': {
                    'branch': self.config.branch.normalization,
                    'trunk': self.config.trunk.normalization
                }
            }
        }
        torch.save(state, path / "transform_state.pt")

    @classmethod
    def load(cls, saved_path: Path, device: str):
        """Matches your testing requirements"""
        state = torch.load(saved_path / "transform_state.pt", map_location=device)
        # Rebuild config from saved state
        config = TransformConfig(
            branch=ComponentTransformConfig(
                normalization=state['config']['normalization']['branch'],
                feature_expansion=FeatureExpansionConfig(
                    **state['config']['branch_feature_expansion']
                ) if state['config']['branch_feature_expansion'] else None,
            ),
            trunk=ComponentTransformConfig(
                normalization=state['config']['normalization']['trunk'],
                feature_expansion=FeatureExpansionConfig(
                    **state['config']['trunk_feature_expansion']
                ) if state['config']['trunk_feature_expansion'] else None
            ),
            output_normalization=None  # Per your current setup
        )._set_device(device, eval(f"torch.{state['config']['dtype']}"))
        
        pipeline = cls(config)
        pipeline.branch_stats = state['branch_stats']
        pipeline.trunk_stats = state['trunk_stats']
        return pipeline

    def inverse_transform(self, component: Literal["branch", "trunk"], tensor: torch.Tensor) -> torch.Tensor:
        """Reverse normalization and expansion in correct order"""
        # Reverse feature expansion first (if applied)
        if component in self.expansion_factors:
            tensor = self.inverse_expansion(component, tensor)
        
        # Reverse normalization if configured
        norm_type = getattr(self.config, component).normalization
        stats = self.branch_stats if component == "branch" else self.trunk_stats
        
        if not norm_type or not stats:
            return tensor
        
        if norm_type == "standardize":
            return tensor * stats['std'] + stats['mean']
        elif norm_type == "minmax_0_1":
            return tensor * (stats['max'] - stats['min']) + stats['min']
        elif norm_type == "minmax_-1_1":
            return (tensor + 1) * (stats['max'] - stats['min']) / 2 + stats['min']
        
        return tensor
    
    def inverse_expansion(self, component: str, x: torch.Tensor) -> torch.Tensor:
        """Reverses feature expansion by slicing"""
        factors = self.expansion_factors.get(component)
        if not factors: return x
        
        # Slice to original dimensions
        return x[..., :factors['original_dim']]
    
    def _update_stats(self, component: str, data: torch.Tensor):
        """Centralized stats calculation"""
        stats = {
            'mean': data.mean(dim=0).detach(),
            'std': data.std(dim=0).detach(),
            'min': data.min(dim=0).detach(), # type: ignore
            'max': data.max(dim=0).detach() # type: ignore
        }
        if component == 'branch':
            self.branch_stats.update(stats)
        else:
            self.trunk_stats.update(stats)