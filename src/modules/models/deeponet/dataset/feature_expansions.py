import torch
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from collections.abc import Callable

class FeatureExpansionRegistry:
    """
    A registry for managing and retrieving feature expansion functions.

    This class provides a decorator to register new expansion functions and a
    method to retrieve a wrapped function that includes the expansion size.
    """
    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a function under a given name.
        """
        def decorator(func):
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get_expansion_fn(cls, name: str, size: int) -> Callable:
        """
        Retrieves a feature expansion function from the registry.

        Args:
            name (str): The name of the registered expansion function.
            size (int): The size parameter for the expansion.

        Returns:
            Callable: A wrapped function that takes a tensor and applies the
                      expansion with the given size.
        """
        def wrapped(x: torch.Tensor):
            return cls._registry[name](x, size)
        return wrapped

@dataclass
class FeatureExpansionConfig:
    """
    Configuration for a feature expansion method.

    Attributes:
        type (Optional[Literal["cosine", "polynomial"]]): The type of expansion.
        size (Optional[int]): The size parameter for the expansion (e.g., number of
                              polynomial terms or frequencies).
        original_dim (Optional[int]): The original dimension of the data before expansion.
    """
    type: Optional[Literal["cosine", "polynomial"]] = None
    size: Optional[int] = None
    original_dim: Optional[int] = None

    def __post_init__(self):
        if self.type and not self.size:
            raise ValueError("Feature expansion requires 'size' when type is specified")
        if self.size and self.size <= 0:
            raise ValueError("Feature expansion size must be > 0")

@FeatureExpansionRegistry.register("sin_cos")
def sin_cos_encoding(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    Applies a sine-cosine encoding.

    Args:
        x (torch.Tensor): The input tensor.
        size (int): The number of sine-cosine pairs to generate.

    Returns:
        torch.Tensor: The expanded tensor.
    """
    features = [x]
    for k in range(1, size+1):
        features.append(torch.cos(k * torch.pi * x))
        features.append(torch.sin(k * torch.pi * x))
    return torch.cat(features, dim=-1)

@FeatureExpansionRegistry.register("polynomial")
def poly_encoding(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    Applies a polynomial feature expansion.

    Args:
        x (torch.Tensor): The input tensor.
        size (int): The highest degree of the polynomial terms.

    Returns:
        torch.Tensor: The expanded tensor.
    """
    return torch.cat([x ** i for i in range(1, size+1)], dim=-1)