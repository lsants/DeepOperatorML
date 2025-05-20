from typing import Any, Type
from collections.abc import Callable
from .config import ComponentConfig

class ComponentRegistry:
    _registry: dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[..., Any]:
        def decorator(component_class) -> Any:
            cls._registry[name] = component_class
            return component_class
        return decorator
    
    @classmethod
    def create(cls, name: str, config: ComponentConfig):
        return cls._registry[name](config)