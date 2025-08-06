from src.modules.models.deeponet.components.registry import ComponentRegistry
from src.modules.models.deeponet.components.branch.chebyshev_kan_branch import ChebyshevKANBranch
from src.modules.models.deeponet.components.branch.matrix_branch import MatrixBranch
from src.modules.models.deeponet.components.branch.mlp_branch import MLPBranch
from src.modules.models.deeponet.components.branch.resnet_branch import ResNetBranch

ComponentRegistry.register(component_type="matrix")(MatrixBranch)
ComponentRegistry.register(component_type='branch_neural', architecture="mlp")(MLPBranch)
ComponentRegistry.register(component_type='branch_neural', architecture="resnet")(ResNetBranch)
ComponentRegistry.register(component_type='branch_neural', architecture="chebychev_kan")(ChebyshevKANBranch)