import numpy as np
import torch
from tqdm.auto import tqdm


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


def train(train_dataloader, model, loss_fn, optimizer, device, val_dataloader=None):
    model.to(device, dtype=torch.float64)
    loss_fn.to(device)
    model.train()
    

    # Training
    total_train_loss = 0.0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device, dtype=torch.float64), y.to(device, dtype=torch.float64)

        # Forward pass
        y_pred = model(X)
        y_pred = torch.squeeze(y_pred)
        loss = loss_fn(y_pred, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)

    # Validation
    if val_dataloader is not None:
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                X_val, y_val = X_val.to(device, dtype=torch.float64), y_val.to(device, dtype=torch.float64)
                y_pred_val = model(X_val)
                y_pred_val = torch.squeeze(y_pred_val)
                val_loss = loss_fn(y_pred_val, y_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        return avg_train_loss, avg_val_loss

    return avg_train_loss


def test(dataloader, model, device, metric='RMSE', custom=0.1):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    total_metric = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device, dtype=torch.float64), y.to(device, dtype=torch.float64)
            y_pred = model(X)
            y_pred = np.squeeze(y_pred)

            match metric:
                case 'MAE':
                    criterion = torch.nn.L1Loss()
                    batch_metric = criterion(y_pred, y).item()
                case 'RMSE':
                    criterion = torch.nn.MSELoss()
                    batch_metric = torch.sqrt(criterion(y_pred, y)).item()
                    print(f'{metric} for batch: {batch_metric}')
            total_metric += batch_metric

    average_metric = total_metric / num_batches
    print(f"Performance ({metric}): {(average_metric):>8e}")
    return average_metric
