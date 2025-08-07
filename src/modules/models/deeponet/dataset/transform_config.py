from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional, Literal, Dict
from src.modules.models.deeponet.dataset.feature_expansions import FeatureExpansionConfig

@dataclass
class ComponentTransformConfig:
    """
    Configuration for a single component's data transformation pipeline.

    Attributes:
        original_dim (Optional[int]): The original dimension of the data before
                                      any feature expansion.
        normalization (Optional[Literal["minmax_0_1",
                                    "minmax_-1_1", "standardize"]]):
                                      The type of normalization to apply.
        feature_expansion (Optional[FeatureExpansionConfig]): The configuration
                                                              for feature expansion.
    """
    original_dim: Optional[int] = None
    normalization: Optional[Literal["minmax_0_1",
                                    "minmax_-1_1", "standardize"]] = None
    feature_expansion: Optional[FeatureExpansionConfig] = None


@dataclass
class TransformConfig:
    """
    Holds the complete configuration for all data transformations in the pipeline.

    This dataclass contains the transformation configurations for the branch,
    trunk, and target components, as well as the global device and data type.
    """
    branch: ComponentTransformConfig
    trunk: ComponentTransformConfig
    target: ComponentTransformConfig
    device: str | torch.device
    dtype: torch.dtype

    @classmethod
    def from_train_config(cls,
                          branch_transforms: Dict,
                          trunk_transforms: Dict,
                          target_transforms: Dict,
                          device: str | torch.device,
                          dtype: torch.dtype):
        """
        Creates a TransformConfig instance from a training configuration dictionary.

        This method parses dictionaries containing transformation settings to
        build the structured configuration object. It correctly handles cases
        where feature expansion might not be specified.

        Args:
            branch_transforms (Dict): Dictionary with branch transformation settings.
            trunk_transforms (Dict): Dictionary with trunk transformation settings.
            target_transforms (Dict): Dictionary with target transformation settings.
            device (str | torch.device): The device to use for tensors.
            dtype (torch.dtype): The data type for tensors.

        Returns:
            TransformConfig: The constructed configuration object.
        """

        branch_expansion_type, branch_expansion_size = None, None
        trunk_expansion_type, trunk_expansion_size = None, None
        if branch_transforms['feature_expansion'] is not None:
            branch_expansion_type = branch_transforms['feature_expansion'].get(
                'type')
            branch_expansion_size = branch_transforms['feature_expansion'].get(
                'size')
        if trunk_transforms['feature_expansion'] is not None:
            trunk_expansion_type = trunk_transforms['feature_expansion'].get(
                'type')
            trunk_expansion_size = trunk_transforms['feature_expansion'].get(
                'size')

        return cls(
            branch=ComponentTransformConfig(normalization=branch_transforms['normalization'],
                                            feature_expansion=FeatureExpansionConfig(type=branch_expansion_type,
                                                                                     size=branch_expansion_size)),
            trunk=ComponentTransformConfig(normalization=trunk_transforms['normalization'],
                                           feature_expansion=FeatureExpansionConfig(type=trunk_expansion_type,
                                                                                    size=trunk_expansion_size)),
            target=ComponentTransformConfig(**target_transforms),
            device=device,
            dtype=dtype
        )

    @classmethod
    def from_exp_config(cls,
                        branch_transforms: Dict,
                        trunk_transforms: Dict,
                        target_transforms: Dict,
                        device: str | torch.device,
                        dtype: torch.dtype):
        """
        Creates a TransformConfig instance from an experiment configuration dictionary.

        This method is similar to `from_train_config` but includes the
        `original_dim` for each component, which is typically saved after a model
        has been trained.

        Args:
            branch_transforms (Dict): Dictionary with branch transformation settings.
            trunk_transforms (Dict): Dictionary with trunk transformation settings.
            target_transforms (Dict): Dictionary with target transformation settings.
            device (str | torch.device): The device to use for tensors.
            dtype (torch.dtype): The data type for tensors.

        Returns:
            TransformConfig: The constructed configuration object.
        """
        branch_expansion_type, branch_expansion_size = None, None
        trunk_expansion_type, trunk_expansion_size = None, None
        if branch_transforms['feature_expansion'] is not None:
            branch_expansion_type = branch_transforms['feature_expansion'].get('type')
            branch_expansion_size = branch_transforms['feature_expansion'].get('size')
        if trunk_transforms['feature_expansion'] is not None:
            trunk_expansion_type = trunk_transforms['feature_expansion'].get('type')
            trunk_expansion_size = trunk_transforms['feature_expansion'].get('size')

        return cls(
            branch=ComponentTransformConfig(
                original_dim=branch_transforms['original_dim'],
                normalization=branch_transforms['normalization'],
                feature_expansion=FeatureExpansionConfig(
                                                type=branch_expansion_type,
                                                size=branch_expansion_size
                            )
                        ),
            trunk=ComponentTransformConfig(
                original_dim=trunk_transforms['original_dim'],
                normalization=trunk_transforms['normalization'],
                feature_expansion=FeatureExpansionConfig(
                    type=trunk_expansion_type,
                    size=trunk_expansion_size
                            )
                        ),
            target=ComponentTransformConfig(**target_transforms),
            device=device,
            dtype=dtype
        )
