from src.modules.model.components.registry import ComponentRegistry
from src.modules.model.components.trunk.orthonormal_trunk import OrthonormalTrunk
from src.modules.model.components.trunk.pod_trunk import PODTrunk
from src.modules.model.components.trunk.mlp_trunk import MLPTrunk
from src.modules.model.components.trunk.resnet_trunk import ResNetTrunk
from src.modules.model.components.trunk.chebyshev_kan_trunk import ChebyshevKANTrunk

ComponentRegistry.register(component_type="pod_trunk")(PODTrunk)
ComponentRegistry.register(component_type="orthonormal_trunk")(OrthonormalTrunk)
ComponentRegistry.register(component_type='neural_trunk', architecture="mlp")(MLPTrunk)
ComponentRegistry.register(component_type='neural_trunk', architecture="resnet")(ResNetTrunk)
ComponentRegistry.register(component_type='neural_trunk', architecture="chebychev_kan")(ChebyshevKANTrunk)
