import numpy as np
import torch

def train(dataloader, model, loss_fn, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate((dataloader)):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        y_pred = model(X)
        y_pred = np.squeeze(y_pred)
        loss = loss_fn(y_pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"loss: {loss:>7f} [{current:>5d}]/{size:>5d}]")
        return loss

def test(dataloader, model, loss_fn, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = np.squeeze(y_pred)
            test_loss += loss_fn(y_pred, y).item()
            correct += (torch.sqrt(loss_fn(y_pred,  y))).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # print(f"Test error: \n Accuracy {(100*correct):>0.1f}%, Avg loss: {test_loss:>8e} \n")
    return test_loss, correct
