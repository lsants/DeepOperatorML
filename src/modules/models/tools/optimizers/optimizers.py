import torch.optim as optim
from torch.optim.sgd import SGD
from torch.optim.lbfgs import LBFGS
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.adamax import Adamax
from torch.optim.rmsprop import RMSprop

OPTIMIZER_MAP = {
    "adam": Adam,
    "adamw": AdamW,
    "adamax": Adamax,
    "lbfgs": LBFGS,
    "sgd": SGD,
    "rmsprop": RMSprop,
}

SCHEDULER_MAP = {
    "step": optim.lr_scheduler.StepLR,
    "exponential": optim.lr_scheduler.ExponentialLR,
    # "cosine": optim.lr_scheduler.CosineAnnealingLR,
}