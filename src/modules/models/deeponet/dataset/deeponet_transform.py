from __future__ import annotations
import torch
import numpy as np
import dataclasses
from pathlib import Path
from typing import Literal
from src.modules.models.deeponet.dataset.transform_config import TransformConfig, ComponentTransformConfig
from src.modules.models.deeponet.dataset.feature_expansions import FeatureExpansionConfig, FeatureExpansionRegistry

class DeepONetTransformPipeline:
    """
    A pipeline for preprocessing data for a DeepONet model.

    This class handles the normalization, feature expansion, and inverse
    transformation of data for the branch, trunk, and target components. It
    can fit statistics from training data, apply transformations, and
    revert them for post-processing results.
    """
    def __init__(self, config: TransformConfig):
        """
        Initializes the DeepONetTransformPipeline.

        Args:
            config (TransformConfig): Configuration object specifying the
                                      transformation types, device, and data type.
        """
        self.config = config
        self.branch_stats: dict = {}
        self.trunk_stats: dict = {}
        self.target_stats: dict = {}
        self.dimension_info: dict = {"trunk": None, "branch": None, "target": None}

    def fit_branch(self, branch_data: np.ndarray) -> None:
        """
        Fits normalization statistics for the branch data.

        Args:
            branch_data (np.ndarray): The raw branch data used to compute
                                      normalization statistics.
        """
        tensor = self._to_tensor(branch_data)
        self.branch_stats = self._compute_stats(
            data=tensor, norm_type=self.config.branch.normalization)

    def fit_trunk(self, trunk_data: np.ndarray) -> None:
        """
        Fits normalization statistics for the trunk data.

        Args:
            trunk_data (np.ndarray): The raw trunk data used to compute
                                     normalization statistics.
        """
        tensor = self._to_tensor(trunk_data)
        self.trunk_stats = self._compute_stats(
            data=tensor, norm_type=self.config.trunk.normalization)

    def fit_target(self, target_data: np.ndarray) -> None:
        """
        Fits normalization statistics for the target data.

        Args:
            target_data (np.ndarray): The raw target data used to compute
                                      normalization statistics.
        """
        tensor = self._to_tensor(target_data)
        self.target_stats = self._compute_stats(
            data=tensor, norm_type=self.config.target.normalization)

    def _compute_stats(self, data: torch.Tensor, norm_type: str | None) -> dict:
        """
        Computes normalization statistics (mean/std or min/max) for the given data.

        Args:
            data (torch.Tensor): The data tensor to compute statistics from.
            norm_type (str | None): The type of normalization to apply.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        stats = {}
        if norm_type == "standardize":
            stats["mean"] = data.mean(dim=0)
            stats["std"] = data.std(dim=0)
        elif norm_type == "minmax_0_1" or norm_type == "minmax_-1_1":
            stats["min"] = data.min(dim=0).values
            stats["max"] = data.max(dim=0).values
        return stats

    def set_branch_stats(self, stats: dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        Sets the normalization statistics for the branch component from a given dictionary.

        Args:
            stats (Dict[str, np.ndarray | torch.Tensor]): A dictionary of statistics.
        """
        self.branch_stats = self._convert_stats_to_tensor(stats)

    def set_trunk_stats(self, stats: dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        Sets the normalization statistics for the trunk component from a given dictionary.

        Args:
            stats (Dict[str, np.ndarray | torch.Tensor]): A dictionary of statistics.
        """
        self.trunk_stats = self._convert_stats_to_tensor(stats)

    def set_target_stats(self, stats: dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        Sets the normalization statistics for the target component from a given dictionary.

        Args:
            stats (Dict[str, np.ndarray | torch.Tensor]): A dictionary of statistics.
        """
        self.target_stats = self._convert_stats_to_tensor(stats)

    def set_dimension_info(self, component: Literal["branch", "trunk", "target"], original_dim: int) -> None:
        """
        Sets the original dimension of a component before any feature expansion.

        Args:
            component (Literal["branch", "trunk", "target"]): The component to set the dimension for.
            original_dim (int): The original dimension of the data.
        """
        self.dimension_info[component] = original_dim

    def _convert_stats_to_tensor(self, stats: dict) -> dict[str, torch.Tensor]:
        """
        Converts statistics from numpy arrays to tensors and moves them to the correct device.

        Args:
            stats (dict): The dictionary of statistics to convert.

        Returns:
            Dict[str, torch.Tensor]: The converted dictionary of tensors.
        """
        converted = {}
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            converted[k] = v.to(device=self.config.device,
                                dtype=self.config.dtype)
        return converted

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a torch tensor with the specified dtype and device.

        Args:
            data (np.ndarray): The numpy array to convert.

        Returns:
            torch.Tensor: The resulting torch tensor.
        """
        return torch.as_tensor(data, dtype=self.config.dtype).to(self.config.device)

    def transform_branch(self, xb: np.ndarray) -> torch.Tensor:
        """
        Transforms branch data by applying normalization.

        Args:
            xb (np.ndarray): The raw branch data to transform.

        Returns:
            torch.Tensor: The normalized branch data tensor.
        """
        tensor = self._to_tensor(xb)
        self.set_dimension_info(
            component="branch", original_dim=tensor.shape[-1])
        return self._apply_normalization(data=tensor, norm_type=self.config.branch.normalization, stats=self.branch_stats)

    def transform_trunk(self, xt: np.ndarray) -> torch.Tensor:
        """
        Transforms trunk data by applying normalization and feature expansion.

        Args:
            xt (np.ndarray): The raw trunk data to transform.

        Returns:
            torch.Tensor: The transformed trunk data tensor.
        """
        tensor = self._to_tensor(xt)
        tensor = self._apply_normalization(
            data=tensor, norm_type=self.config.trunk.normalization, stats=self.trunk_stats)
        self.set_dimension_info(
            component="trunk", original_dim=tensor.shape[-1])
        return self._apply_expansion(data=tensor, component="trunk")

    def transform_target(self, y: np.ndarray) -> torch.Tensor:
        """
        Transforms target data by applying normalization.

        Args:
            y (np.ndarray): The raw target data to transform.

        Returns:
            torch.Tensor: The normalized target data tensor.
        """
        tensor = self._to_tensor(y)
        self.set_dimension_info(
            component="target", original_dim=tensor.shape[-1])
        return self._apply_normalization(data=tensor, norm_type=self.config.target.normalization, stats=self.target_stats)

    def _apply_normalization(self, data: torch.Tensor, norm_type: str | None, stats: dict) -> torch.Tensor:
        """
        Applies a normalization transformation to the data.

        Args:
            data (torch.Tensor): The tensor to normalize.
            norm_type (str | None): The type of normalization to apply.
            stats (dict): A dictionary of statistics for the normalization.

        Returns:
            torch.Tensor: The normalized tensor.

        Raises:
            RuntimeError: If normalization is requested but no statistics are available.
        """
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
        """
        Applies feature expansion to the data and tracks original dimensions.

        Args:
            data (torch.Tensor): The tensor to expand.
            component (Literal["trunk", "branch", "target"]): The component for which
                                                              to apply expansion.

        Returns:
            torch.Tensor: The expanded tensor.
        """
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
        """
        Reverses the transformations applied to a tensor.

        Args:
            component (Literal["branch", "trunk", "target"]): The component to inverse transform.
            tensor (torch.Tensor): The transformed tensor to revert.

        Returns:
            torch.Tensor: The tensor in its original scale.
        """
        # Reverse expansion first
        if self.dimension_info.get(component):
            tensor = tensor[..., :self.dimension_info[component]]

        # Reverse normalization
        stats = getattr(self, f"{component}_stats")
        norm_type = getattr(self.config, component).normalization
        return self._inverse_normalize(tensor, norm_type, stats)

    def _inverse_normalize(self, data: torch.Tensor, norm_type: str, stats: dict) -> torch.Tensor:
        """
        Reverses a normalization transformation using stored statistics.

        Args:
            data (torch.Tensor): The tensor to inverse normalize.
            norm_type (str): The type of normalization to reverse.
            stats (dict): A dictionary of statistics for the normalization.

        Returns:
            torch.Tensor: The inverse normalized tensor.
        """
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
        """
        Saves the pipeline's state (config, stats, and dimension info) to a file.

        Args:
            path (Path): The directory path to save the state file.
        """
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
        """
        Extracts the state of a single component's configuration.

        Args:
            component (str): The name of the component ("branch", "trunk", or "target").

        Returns:
            dict: A dictionary containing the component's normalization and expansion config.
        """
        cfg = getattr(self.config, component)
        return {
            "normalization": cfg.normalization,
            "feature_expansion": dataclasses.asdict(cfg.feature_expansion) if cfg.feature_expansion else None
        }

    @classmethod
    def load(cls, path: Path, device: str) -> DeepONetTransformPipeline:
        """
        Loads a DeepONetTransformPipeline instance from a saved state.

        Args:
            path (Path): The directory path where the state file is located.
            device (str): The device to map the loaded tensors to.

        Returns:
            DeepONetTransformPipeline: The loaded and re-initialized pipeline instance.
        """
        state = torch.load(path / "transform_state.pt", map_location=device)
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

        pipeline = cls(config)
        pipeline.branch_stats = state["branch_stats"]
        pipeline.trunk_stats = state["trunk_stats"]
        pipeline.target_stats = state["target_stats"]
        pipeline.dimension_info = state["dimension_info"]
        return pipeline
