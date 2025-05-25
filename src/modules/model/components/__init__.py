# components/__init__.py
from typing import TYPE_CHECKING
from .registry import ComponentRegistry  # Explicit export
from .component_factory import BranchFactory, TrunkFactory
__all__ = ['ComponentRegistry', 'BranchFactory', 'TrunkFactory']

if TYPE_CHECKING:
    from .trunk.mlp_trunk import MLPTrunk
    from .trunk.pod_trunk import PODTrunk
