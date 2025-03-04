from .mlp import MLP
from .kan import ChebyshevKAN
from .resnet import ResNet

NETWORK_ARCHITECTURES = {
    'mlp': MLP,
    'kan': ChebyshevKAN,
    'resnet': ResNet
}