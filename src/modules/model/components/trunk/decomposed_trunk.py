from ..config import ComponentConfig
from .trunk_factory import TrunkRegistry
from .base import Trunk


@TrunkRegistry.register('decomposed_trunk')
class DecomposedTrunk(Trunk):
    """Post-decomposition trunk for phase 2"""

    def __init__(self, config: ComponentConfig):
        self.decomposed_trunk = self._get_decomposed_trunk(config)

    def _get_decomposed_trunk(self, config: ComponentConfig):
        pass

    def forward(self, x):
        return x @ self.decomposed_trunk
