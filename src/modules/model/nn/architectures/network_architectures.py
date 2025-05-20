from .mlp import MLP
from .chebyshev_kan import ChebyshevKAN
from .resnet import ResNet

NETWORK_ARCHITECTURES = {
    'mlp': MLP,
    'kan': ChebyshevKAN,
    'resnet': ResNet
}