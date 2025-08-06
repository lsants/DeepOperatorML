from __future__ import annotations
import torch
import numpy as np
import dataclasses
from pathlib import Path
from typing import Literal
from src.modules.models.deeponet.dataset.transform_config import TransformConfig, ComponentTransformConfig
from src.modules.models.deeponet.dataset.feature_expansions import FeatureExpansionConfig, FeatureExpansionRegistry

class DeepONetTransformPipeline:
    def __init__(self, config: TransformConfig):
        self.config = config
        self.branch_stats: dict = {}
        self.trunk_stats: dict = {}
        self.target_stats: dict = {}
        self.dimension_info: dict = {"trunk": None, "branch": None, "target": None}

    def fit_branch(self, branch_data: np.ndarray) -> None:
        tensor = self._to_tensor(branch_data)
        self.branch_stats = self._compute_stats(
            data=tensor, norm_type=self.config.branch.normalization)

    def fit_trunk(self, trunk_data: np.ndarray) -> None:
        tensor = self._to_tensor(trunk_data)
        self.trunk_stats = self._compute_stats(
            data=tensor, norm_type=self.config.trunk.normalization)

    def fit_target(self, target_data: np.ndarray) -> None:
        tensor = self._to_tensor(target_data)
        self.target_stats = self._compute_stats(
            data=tensor, norm_type=self.config.target.normalization)

    def _compute_stats(self, data: torch.Tensor, norm_type: str | None) -> dict:
        stats = {}
        if norm_type == "standardize":
            stats["mean"] = data.mean(dim=0)
            stats["std"] = data.std(dim=0)
        elif norm_type == "minmax_0_1" or norm_type == "minmax_-1_1":
            stats["min"] = data.min(dim=0).values
            stats["max"] = data.max(dim=0).values
        return stats

    def set_branch_stats(self, stats: dict[str, np.ndarray | torch.Tensor]) -> None:
        self.branch_stats = self._convert_stats_to_tensor(stats)

    def set_trunk_stats(self, stats: dict[str, np.ndarray | torch.Tensor]) -> None:
        self.trunk_stats = self._convert_stats_to_tensor(stats)

    def set_target_stats(self, stats: dict[str, np.ndarray | torch.Tensor]) -> None:
        self.target_stats = self._convert_stats_to_tensor(stats)

    def set_dimension_info(self, component: Literal["branch", "trunk", "target"], original_dim: int) -> None:
        self.dimension_info[component] = original_dim

    def _convert_stats_to_tensor(self, stats: dict) -> dict[str, torch.Tensor]:
        converted = {}
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            converted[k] = v.to(device=self.config.device,
                                dtype=self.config.dtype)
        return converted

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, dtype=self.config.dtype).to(self.config.device)

    def transform_branch(self, xb: np.ndarray) -> torch.Tensor:
        tensor = self._to_tensor(xb)
        self.set_dimension_info(
            component="branch", original_dim=tensor.shape[-1])
        return self._apply_normalization(data=tensor, norm_type=self.config.branch.normalization, stats=self.branch_stats)

    def transform_trunk(self, xt: np.ndarray) -> torch.Tensor:
        tensor = self._to_tensor(xt)
        tensor = self._apply_normalization(
            data=tensor, norm_type=self.config.trunk.normalization, stats=self.trunk_stats)
        self.set_dimension_info(
            component="trunk", original_dim=tensor.shape[-1])
        return self._apply_expansion(data=tensor, component="trunk")

    def transform_target(self, y: np.ndarray) -> torch.Tensor:
        tensor = self._to_tensor(y)
        self.set_dimension_info(
            component="target", original_dim=tensor.shape[-1])
        return self._apply_normalization(data=tensor, norm_type=self.config.target.normalization, stats=self.target_stats)

    def _apply_normalization(self, data: torch.Tensor, norm_type: str | None, stats: dict) -> torch.Tensor:
        if norm_type is None:
            return data
        elif norm_type is not None and not stats:
            raise RuntimeError(f"{norm_type} expects non-empty 'stats'")
        if norm_type == "standardize":
            return (data - stats["mean"]) / stats["std"]
        if norm_type == "minmax_0_1":
            return (data - stats["min"]) / (stats["max"] - stats["min"])
        if norm_type == "minmax_-1_1":
            return 2 * (data - stats["min"]) / (stats["max"] - stats["min"]) - 1
        return data

    def _apply_expansion(self, data: torch.Tensor, component: Literal["trunk", "branch", "target"]) -> torch.Tensor:
        """Apply feature expansion and track original dimensions"""
        expansion_cfg = getattr(self.config, component).feature_expansion
        if not expansion_cfg or expansion_cfg.type is None:
            return data
        # Store original dimension on first application
        if self.dimension_info[component] is None:
            self.dimension_info[component] = data.shape[-1]

        expansion_fn = FeatureExpansionRegistry.get_expansion_fn(
            expansion_cfg.type, expansion_cfg.size
        )
        return expansion_fn(data)

    def inverse_transform(self, component: Literal["branch", "trunk", "target"], tensor: torch.Tensor) -> torch.Tensor:
        """Reverse transformations in correct order"""
        # Reverse expansion first
        if self.dimension_info.get(component):
            tensor = tensor[..., :self.dimension_info[component]]

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
        if norm_type == "minmax_0_1":
            return data * (stats["max"] - stats["min"]) + stats["min"]
        if norm_type == "minmax_-1_1":
            return (data + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]

        return data

    def save(self, path: Path) -> None:
        state = {
            "config": {
                "dtype": str(self.config.dtype),
                "device": str(self.config.device),
                "branch": self._component_state("branch"),
                "trunk": self._component_state("trunk"),
                "target": self._component_state("target"),
            },
            "branch_stats": self.branch_stats,
            "trunk_stats": self.trunk_stats,
            "target_stats": self.target_stats,
            "dimension_info": self.dimension_info,
        }
        torch.save(state, path / "transform_state.pt")

    def _component_state(self, component: str) -> dict:
        cfg = getattr(self.config, component)
        return {
            "normalization": cfg.normalization,
            "feature_expansion": dataclasses.asdict(cfg.feature_expansion) if cfg.feature_expansion else None
        }

    @classmethod
    def load(cls, path: Path, device: str) -> DeepONetTransformPipeline:
        state = torch.load(path / "transform_state.pt", map_location=device)

        # Reconstruct config
        config = TransformConfig(
            branch=ComponentTransformConfig(
                normalization=state["config"]["branch"]["normalization"],
                feature_expansion=FeatureExpansionConfig(
                    **state["config"]["branch"]["feature_expansion"])
                if state["config"]["branch"]["feature_expansion"] else None
            ),
            trunk=ComponentTransformConfig(
                normalization=state["config"]["trunk"]["normalization"],
                feature_expansion=FeatureExpansionConfig(
                    **state["config"]["trunk"]["feature_expansion"])
                if state["config"]["trunk"]["feature_expansion"] else None
            ),
            target=ComponentTransformConfig(
                normalization=state["config"]["target"]["normalization"],
                feature_expansion=FeatureExpansionConfig(
                    **state["config"]["target"]["feature_expansion"])
                if state["config"]["target"]["feature_expansion"] else None
            ),
            device=device,
            dtype=state["config"]["dtype"]
        )

        # Create pipeline and restore state
        pipeline = cls(config)
        pipeline.branch_stats = state["branch_stats"]
        pipeline.trunk_stats = state["trunk_stats"]
        pipeline.target_stats = state["target_stats"]
        pipeline.dimension_info = state["dimension_info"]
        return pipeline
