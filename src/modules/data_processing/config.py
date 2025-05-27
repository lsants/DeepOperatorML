from __future__ import annotations
from dataclasses import dataclass, field
import torch
from typing import Optional, Literal, Dict

from src.modules.model.components import branch
from .data_augmentation.feature_expansions import FeatureExpansionConfig

@dataclass
class ComponentTransformConfig:
    normalization: Optional[Literal["minmax_0_1", "minmax_-1_1", "standardize"]] = None
    feature_expansion: Optional[FeatureExpansionConfig] = None

@dataclass
class TransformConfig:
    branch: ComponentTransformConfig
    trunk: ComponentTransformConfig
    device: str | torch.device  # Will be set from TrainConfig
    dtype: torch.dtype  # Will be set from TrainConfig
    target_normalization: Optional[Literal["minmax_0_1", "minmax_-1_1", "standardize"]] = None

    @classmethod
    def from_train_config(cls, 
                          branch_transforms: Dict, 
                          trunk_transforms: Dict, 
                          target_transforms: Dict, 
                          device: str | torch.device, 
                          dtype: torch.dtype):
        target_normalization = None
        if target_transforms is not None:
            target_normalization = target_transforms.get("normalization")
        
        branch_expansion_type, branch_expansion_size = None, None
        trunk_expansion_type, trunk_expansion_size = None, None
        if branch_transforms['feature_expansion'] is not None:
            branch_expansion_type = branch_transforms['feature_expansion'].get('type')
            branch_expansion_size = branch_transforms['feature_expansion'].get('size')
        if trunk_transforms['feature_expansion'] is not None:
            trunk_expansion_type = trunk_transforms['feature_expansion'].get('type')
            trunk_expansion_size = trunk_transforms['feature_expansion'].get('size')
            
        return cls(
            branch=ComponentTransformConfig(normalization=branch_transforms['normalization'],
                                            feature_expansion=FeatureExpansionConfig(type=branch_expansion_type,
                                                                                     size=branch_expansion_size)),
            trunk=ComponentTransformConfig(normalization=trunk_transforms['normalization'],
                                            feature_expansion=FeatureExpansionConfig(type=trunk_expansion_type,
                                                                                     size=trunk_expansion_size)),
            target_normalization=target_normalization, 
            device=device, 
            dtype=dtype
        )
