from dataclasses import dataclass, field
import torch
from typing import Optional, Literal, Dict
from .data_augmentation.feature_expansions import FeatureExpansionConfig

@dataclass
class ComponentTransformConfig:
    normalization: Optional[Literal["minmax_0_1", "minmax_-1_1", "standardize"]] = None
    feature_expansion: Optional[FeatureExpansionConfig] = None

@dataclass
class TransformConfig:
    branch: ComponentTransformConfig
    trunk: ComponentTransformConfig
    output_normalization: Optional[Literal["minmax_0_1", "minmax_-1_1", "standardize"]] = None
    device: str = field(init=False)  # Will be set from TrainConfig
    dtype: torch.dtype = field(init=False)  # Will be set from TrainConfig

    @classmethod
    def from_train_config(cls, train_cfg: Dict, device: str, dtype: torch.dtype):
        return cls(
            branch=ComponentTransformConfig(**train_cfg["transforms"]["branch"]),
            trunk=ComponentTransformConfig(**train_cfg["transforms"]["trunk"]),
            output_normalization=train_cfg["transforms"].get("output_normalization")
        )._set_device(device, dtype)

    def _set_device(self, device: str, dtype: torch.dtype) -> 'TransformConfig':
        self.device = device
        self.dtype = dtype
        return self
