import torch
from .registry import ComponentRegistry
from .trunk.config import TrunkConfig, TrunkConfigValidator
from .branch.config import BranchConfig, BranchConfigValidator
from typing import TypeVar
import inspect
from typing import TYPE_CHECKING
from .trunk.orthonormal_trunk import OrthonormalTrunk
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
        component_class, _ = ComponentRegistry.get(
            component_type=config.component_type,
            architecture=config.architecture
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
        if config.component_type == "orthonormal_trunk":
            inner_trunk = cls.build(config.inner_config)  # type: ignore
            basis_tensor = torch.as_tensor(config.T_matrix)
            return OrthonormalTrunk(inner_trunk, basis_tensor)
        
        if config.component_type == "pod_trunk":
            return PODTrunk(inner_trunk, basis_tensor)

        TrunkConfigValidator.validate(config)
        # Get component class
        component_class, _ = ComponentRegistry.get(
            component_type=config.component_type,
            architecture=config.architecture
        )
        # Filter valid constructor parameters
        sig = inspect.signature(component_class.__init__)
        valid_params = {
            k: v for k, v in config.__dict__.items()
            if k in sig.parameters and k != "self"
        }

        return component_class(**valid_params)
