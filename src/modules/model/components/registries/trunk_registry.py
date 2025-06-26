from ..registry import ComponentRegistry
from ..trunk.orthonormal_trunk import OrthonormalTrunk
from ..trunk.pod_trunk import PODTrunk
from ..trunk.mlp_trunk import MLPTrunk
from ..trunk.resnet_trunk import ResNetTrunk
from ..trunk.chebyshev_kan_trunk import ChebyshevKANTrunk

ComponentRegistry.register(component_type="pod_trunk")(PODTrunk)
ComponentRegistry.register(
    component_type="orthonormal_trunk")(OrthonormalTrunk)
ComponentRegistry.register(
    component_type='neural_trunk', architecture="mlp")(MLPTrunk)
ComponentRegistry.register(
    component_type='neural_trunk', architecture="resnet")(ResNetTrunk)
ComponentRegistry.register(
    component_type='neural_trunk', architecture="chebychev_kan")(ChebyshevKANTrunk)
