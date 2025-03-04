import torch

ACTIVATION_MAP = {
    'relu': torch.nn.ReLU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'elu': torch.nn.ELU(),
    'gelu': torch.nn.GELU(),
    'softplus': torch.nn.Softplus(),
    'identity': torch.nn.Identity()
}