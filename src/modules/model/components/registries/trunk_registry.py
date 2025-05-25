from ..registry import ComponentRegistry
from ..trunk.decomposed_trunk import DecomposedTrunk
from ..trunk.pod_trunk import PODTrunk
from ..trunk.mlp_trunk import MLPTrunk
from ..trunk.resnet_trunk import ResNetTrunk
from ..trunk.chebyshev_kan_trunk import ChebyshevKANTrunk

ComponentRegistry.register(component_type="pod")(PODTrunk)
ComponentRegistry.register(component_type="decomposed")(DecomposedTrunk)
ComponentRegistry.register(component_type='trunk_neural', architecture="mlp")(MLPTrunk)
ComponentRegistry.register(component_type='trunk_neural', architecture="resnet")(ResNetTrunk)
ComponentRegistry.register(component_type='trunk_neural', architecture="chebychev_kan")(ChebyshevKANTrunk)