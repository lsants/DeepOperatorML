import torch.optim as optim

OPTIMIZER_MAP = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}

SCHEDULER_MAP = {
    "step": optim.lr_scheduler.StepLR,
    "exponential": optim.lr_scheduler.ExponentialLR,
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
}