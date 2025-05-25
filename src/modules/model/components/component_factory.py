import torch
from .registry import ComponentRegistry
from .trunk.config import TrunkConfig, TrunkConfigValidator
from .branch.config import BranchConfig, BranchConfigValidator
from typing import Type, Dict, TypeVar, Tuple, Optional
import inspect
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .trunk.config import TrunkConfig
    from .branch.config import BranchConfig

T = TypeVar('T')

class BranchFactory:
    @classmethod
    def build(cls, config: BranchConfig) -> torch.nn.Module:
        # Validate config first
        BranchConfigValidator.validate(config)
        # Get component class
        if config.component_type == "branch_neural":
            component_class, _ = ComponentRegistry.get(
                component_type="branch_neural",
                architecture=config.architecture
            )
        else:
            component_class, _ = ComponentRegistry.get(
                component_type=config.component_type,
                architecture=None
            )
        # Filter valid constructor parameters
        sig = inspect.signature(component_class.__init__)
        valid_params = {
            k: v for k, v in config.__dict__.items()
            if k in sig.parameters and k != "self"
        }

        return component_class(**valid_params)

class TrunkFactory:
    @classmethod
    def build(cls, config: TrunkConfig) -> torch.nn.Module:
        # Validate config first
        TrunkConfigValidator.validate(config)
        # Get component class
        if config.component_type == "trunk_neural":
            component_class, _ = ComponentRegistry.get(
                component_type="trunk_neural",
                architecture=config.architecture
            )
        else:
            component_class, _ = ComponentRegistry.get(
                component_type=config.component_type,
                architecture=None
            )

        # Filter valid constructor parameters
        sig = inspect.signature(component_class.__init__)
        valid_params = {
            k: v for k, v in config.__dict__.items()
            if k in sig.parameters and k != "self"
        }

        return component_class(**valid_params)
