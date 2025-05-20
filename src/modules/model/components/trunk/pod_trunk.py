from .base import Trunk
from .trunk_factory import TrunkRegistry
from ..config import ComponentConfig
from .base import Trunk

@TrunkRegistry.register("pod_trunk")
class PODTrunk(Trunk):
    def __init__(self, config: ComponentConfig):
        self.basis = self._compute_pod_basis(var_share=config.params['var_share'])

    def _compute_pod_basis(self, var_share: float):
        pass