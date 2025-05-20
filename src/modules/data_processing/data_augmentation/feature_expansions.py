import torch
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from collections.abc import Callable
# --------------------------
# Feature Expansion Registry
# --------------------------
class FeatureExpansionRegistry:
    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(func):
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get_expansion_fn(cls, name: str, size: int) -> Callable:
        def wrapped(x: torch.Tensor):
            return cls._registry[name](x, size)
        return wrapped

# --------------------------
# Core Transformation Classes
# --------------------------
@dataclass
class FeatureExpansionConfig:
    type: Optional[Literal["cosine", "polynomial"]] = None
    size: Optional[int] = None

    def __post_init__(self):
        if self.type and not self.size:
            raise ValueError("Feature expansion requires 'size' when type is specified")
        if self.size and self.size <= 0:
            raise ValueError("Feature expansion size must be > 0")

@FeatureExpansionRegistry.register("cosine")
def cosine_expansion(x: torch.Tensor, size: int) -> torch.Tensor:
    features = [x]
    for k in range(1, size+1):
        features.append(torch.cos(k * torch.pi * x))
    return torch.cat(features, dim=-1)

@FeatureExpansionRegistry.register("polynomial")
def poly_expansion(x: torch.Tensor, size: int) -> torch.Tensor:
    return torch.cat([x ** i for i in range(1, size+1)], dim=-1)