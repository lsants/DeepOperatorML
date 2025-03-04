import torch.optim as optim

OPTIMIZER_MAP = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}