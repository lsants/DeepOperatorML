from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional, Literal, Dict
from src.modules.models.deeponet.dataset.feature_expansions import FeatureExpansionConfig

@dataclass
class ComponentTransformConfig:
    original_dim: Optional[int] = None
    normalization: Optional[Literal["minmax_0_1",
                                    "minmax_-1_1", "standardize"]] = None
    feature_expansion: Optional[FeatureExpansionConfig] = None


@dataclass
class TransformConfig:
    branch: ComponentTransformConfig
    trunk: ComponentTransformConfig
    target: ComponentTransformConfig
    device: str | torch.device  # Will be set from TrainConfig
    dtype: torch.dtype  # Will be set from TrainConfig

    @classmethod
    def from_train_config(cls,
                          branch_transforms: Dict,
                          trunk_transforms: Dict,
                          target_transforms: Dict,
                          device: str | torch.device,
                          dtype: torch.dtype):

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
            branch=ComponentTransformConfig(original_dim=branch_transforms['original_dim'],
                                            normalization=branch_transforms['normalization'],
                                            feature_expansion=FeatureExpansionConfig(type=branch_expansion_type,
                                                                                     size=branch_expansion_size)),
            trunk=ComponentTransformConfig(original_dim=trunk_transforms['original_dim'],
                                           normalization=trunk_transforms['normalization'],
                                           feature_expansion=FeatureExpansionConfig(type=trunk_expansion_type,
                                                                                    size=trunk_expansion_size)),
            target=ComponentTransformConfig(**target_transforms),
            device=device,
            dtype=dtype
        )
