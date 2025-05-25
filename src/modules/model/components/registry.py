from typing import Dict, Tuple, Optional, Type
import torch
import inspect

class ComponentRegistry:
    """Handles registration/retrieval of components with composite keys (type, architecture)"""
    _registry: Dict[Tuple[str, Optional[str]], Type[torch.nn.Module]] = {}

    @classmethod
    def register(cls,
                 component_type: str,
                 architecture: Optional[str] = None
                 ):
        """Decorator for registering components with type + optional architecture"""
        def decorator(component_class: Type[torch.nn.Module]):
            key = (component_type, architecture)
            if key in cls._registry:
                raise ValueError(f"Component {key} already registered")
            cls._registry[key] = {
                "class": component_class,
                "required_params": list(inspect.signature(component_class.__init__).parameters.keys())
            }
            return component_class
        return decorator

    @classmethod
    def get(
        cls,
        component_type: str,
        architecture: Optional[str] = None
    ) -> Tuple[Type[torch.nn.Module], list[str]]:  # Return tuple instead of dict
        key = (component_type, architecture)
        if key not in cls._registry:
            raise ValueError(f"Component {key} not registered")
            
        entry = cls._registry[key]
        return entry["class"], entry["required_params"]  # Explicit tuple return