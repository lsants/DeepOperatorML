from ..registry import ComponentRegistry
from ..branch.chebyshev_kan_branch import ChebyshevKANBranch
from ..branch.matrix_branch import MatrixBranch
from ..branch.mlp_branch import MLPBranch
from ..branch.resnet_branch import ResNetBranch

ComponentRegistry.register(component_type="matrix")(MatrixBranch)
ComponentRegistry.register(component_type='branch_neural', architecture="mlp")(MLPBranch)
ComponentRegistry.register(component_type='branch_neural', architecture="resnet")(ResNetBranch)
ComponentRegistry.register(component_type='branch_neural', architecture="chebychev_kan")(ChebyshevKANBranch)