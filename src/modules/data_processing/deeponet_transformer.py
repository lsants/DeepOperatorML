from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Literal
from collections.abc import Callable
from .config import TransformConfig, ComponentTransformConfig
from .data_augmentation.feature_expansions import FeatureExpansionConfig, FeatureExpansionRegistry

class DeepONetTransformPipeline:
    def __init__(self, config: TransformConfig):
        self.config = config
        self.branch_stats: dict = {}
        self.trunk_stats: dict = {}
        self.target_stats: dict = {}
        self.expansion_info: dict = {"trunk": None, "branch": None}

    def fit_branch(self, branch_data: np.ndarray) -> None:
        """Compute statistics for branch normalization"""
        tensor = self._to_tensor(branch_data)
        self.branch_stats = self._compute_stats(data=tensor, norm_type=self.config.branch.normalization)

    def fit_trunk(self, trunk_data: np.ndarray) -> None:
        """Compute statistics for trunk normalization"""
        tensor = self._to_tensor(trunk_data)
        self.trunk_stats = self._compute_stats(data=tensor, norm_type=self.config.trunk.normalization)

    def fit_target(self, target_data: np.ndarray) -> None:
        """Compute statistics for target normalization"""
        tensor = self._to_tensor(target_data)
        self.target_stats = self._compute_stats(data=tensor, norm_type=self.config.target_normalization)

    def _compute_stats(self, data: torch.Tensor, norm_type: str | None) -> dict:
        """Compute and return relevant statistics for normalization type"""
        stats = {}
        if norm_type == "standardize":
            stats["mean"] = data.mean(dim=0)
            stats["std"] = data.std(dim=0)
        elif norm_type == "0_1_min_max":
            stats["min"] = data.min(dim=0).values
            stats["max"] = data.max(dim=0).values
        elif norm_type == "-1_1_min_max":
            stats["min"] = data.min(dim=0).values
            stats["max"] = data.max(dim=0).values
        return stats

    def set_branch_stats(self, stats: dict[str, np.ndarray | torch.Tensor]) -> None:
        """Set precomputed branch statistics directly"""
        self.branch_stats = self._convert_stats_to_tensor(stats)

    def set_trunk_stats(self, stats: dict[str, np.ndarray | torch.Tensor]) -> None:
        """Set precomputed trunk statistics directly"""
        self.trunk_stats = self._convert_stats_to_tensor(stats)

    def set_target_stats(self, stats: dict[str, np.ndarray | torch.Tensor]) -> None:
        """Set precomputed target statistics directly"""
        self.target_stats = self._convert_stats_to_tensor(stats)

    def set_expansion_info(self, component: Literal["branch", "trunk"], original_dim: int) -> None:
        """Set precomputed expansion information"""
        self.expansion_info[component] = original_dim

    def _convert_stats_to_tensor(self, stats: dict) -> dict[str, torch.Tensor]:
        """Convert numpy arrays to properly configured tensors"""
        converted = {}
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            converted[k] = v.to(device=self.config.device, dtype=self.config.dtype)
        return converted

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, dtype=self.config.dtype).to(self.config.device)

    def transform_branch(self, xb: np.ndarray) -> torch.Tensor:
        """Apply branch normalization"""
        tensor = self._to_tensor(xb)
        return self._apply_normalization(data=tensor, norm_type=self.config.branch.normalization, stats=self.branch_stats)

    def transform_trunk(self, xt: np.ndarray) -> torch.Tensor:
        """Apply trunk normalization and feature expansion"""
        tensor = self._to_tensor(xt)
        tensor = self._apply_normalization(data=tensor, norm_type=self.config.trunk.normalization, stats=self.trunk_stats)
        return self._apply_expansion(data=tensor, component="trunk")

    def transform_target(self, y: np.ndarray) -> torch.Tensor:
        """Apply target normalization"""
        tensor = self._to_tensor(y)
        return self._apply_normalization(data=tensor, norm_type=self.config.target_normalization, stats=self.target_stats)

    def _apply_normalization(self, data: torch.Tensor, norm_type: str | None, stats: dict) -> torch.Tensor:
        """Apply normalization using precomputed statistics"""
        if norm_type is None or not stats:
            return data
            
        if norm_type == "standardize":
            return (data - stats["mean"]) / stats["std"]
        if norm_type == "0_1_min_max":
            return (data - stats["min"]) / (stats["max"] - stats["min"])
        if norm_type == "-1_1_min_max":
            return 2 * (data - stats["min"]) / (stats["max"] - stats["min"]) - 1
        return data

    def _apply_expansion(self, data: torch.Tensor, component: Literal["trunk", "branch"]) -> torch.Tensor:
        """Apply feature expansion and track original dimensions"""
        expansion_cfg = getattr(self.config, component).feature_expansion
        if not expansion_cfg:
            return data

        # Store original dimension on first application
        if self.expansion_info[component] is None:
            self.expansion_info[component] = data.shape[-1]

        expansion_fn = FeatureExpansionRegistry.get_expansion_fn(
            expansion_cfg.type, expansion_cfg.size
        )
        return expansion_fn(data)

    def inverse_transform(self, component: Literal["branch", "trunk"], tensor: torch.Tensor) -> torch.Tensor:
        """Reverse transformations in correct order"""
        # Reverse expansion first
        if self.expansion_info.get(component):
            tensor = tensor[..., :self.expansion_info[component]]
            
        # Reverse normalization
        stats = getattr(self, f"{component}_stats")
        norm_type = getattr(self.config, component).normalization
        return self._inverse_normalize(tensor, norm_type, stats)

    def _inverse_normalize(self, data: torch.Tensor, norm_type: str, stats: dict) -> torch.Tensor:
        """Reverse normalization using stored statistics"""
        if not norm_type or not stats:
            return data
            
        if norm_type == "standardize":
            return data * stats["std"] + stats["mean"]
        if norm_type == "0_1_min_max":
            return data * (stats["max"] - stats["min"]) + stats["min"]
        if norm_type == "-1_1_min_max":
            return (data + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        return data

    def save(self, path: Path) -> None:
        """Save transformation state"""
        state = {
            "config": {
                "dtype": self.config.dtype,
                "device": self.config.device,
                "branch": self._component_state("branch"),
                "trunk": self._component_state("trunk"),
                "target": {"normalization": self.config.target_normalization},
            },
            "branch_stats": self.branch_stats,
            "trunk_stats": self.trunk_stats,
            "target_stats": self.target_stats,
            "expansion_info": self.expansion_info,
        }
        torch.save(state, path / "transform_state.pt")

    def _component_state(self, component: str) -> dict:
        cfg = getattr(self.config, component)
        return {
            "normalization": cfg.normalization,
            "feature_expansion": cfg.feature_expansion if cfg.feature_expansion else None
        }

    @classmethod
    def load(cls, path: Path, device: str) -> DeepONetTransformPipeline:
        """Load transformation pipeline from saved state"""
        state = torch.load(path / "transform_state.pt", map_location=device)
        
        # Reconstruct config
        config = TransformConfig(
            branch=ComponentTransformConfig(
                normalization=state["config"]["branch"]["normalization"],
                feature_expansion=FeatureExpansionConfig(**state["config"]["branch"]["feature_expansion"])
                if state["config"]["branch"]["feature_expansion"] else None
            ),
            trunk=ComponentTransformConfig(
                normalization=state["config"]["trunk"]["normalization"],
                feature_expansion=FeatureExpansionConfig(**state["config"]["trunk"]["feature_expansion"])
                if state["config"]["trunk"]["feature_expansion"] else None
            ),
            target_normalization=state["config"]["target"]["normalization"],
            device=device,
            dtype=state["config"]["dtype"]
        )

        # Create pipeline and restore state
        pipeline = cls(config)
        pipeline.branch_stats = state["branch_stats"]
        pipeline.trunk_stats = state["trunk_stats"]
        pipeline.target_stats = state["target_stats"]
        pipeline.expansion_info = state["expansion_info"]
        return pipeline