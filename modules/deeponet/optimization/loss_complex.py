import torch

def loss_complex(targets, preds):
    loss = 0
    for target, pred in zip(targets, preds):
         loss += torch.mean((pred - target) ** 2)

    return loss