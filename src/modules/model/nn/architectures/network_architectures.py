from .mlp import MLP
from .chebyshev_kan import ChebyshevKAN
from .jacobi_kan import JacobiKAN
from .legendre_kan import LegendreKAN
from .resnet import ResNet

NETWORK_ARCHITECTURES = {
    'mlp': MLP,
    'chebysev_kan': ChebyshevKAN,
    'jacobi_kan': JacobiKAN,
    'legendre_kan': ChebyshevKAN,
    'resnet': ResNet
    # future: 'cnn': CNN,
    # future: 'transformer': Transformer,
}

