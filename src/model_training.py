import numpy as np
import torch

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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


def test(dataloader, model, loss_fn, device='cpu', metric='RMSE'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    total_test_loss, total_metric = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = np.squeeze(y_pred)

            match metric:
                case 'MAE':
                    criterion = torch.nn.L1Loss()
                    batch_metric = criterion(y_pred, y).item()
                case 'RMSE':
                    criterion = torch.nn.MSELoss()
                    batch_metric = torch.sqrt(criterion(y_pred, y)).item()
            total_test_loss += loss_fn(y_pred, y).item()
            total_metric += batch_metric

    avg_test_loss = total_test_loss / num_batches
    average_metric = total_metric / size
    # print(f"Test error: \n Epoch performance ({metric}) {(average_metric):>8e}, loss for batch: {test_loss:>8e} \n")
    return avg_test_loss, average_metric
