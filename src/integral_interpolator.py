# ------------ Integration by Interpolation ---------------
''' 
We want to use a NN to compute the integral I of a polynomial function in the form of:
    αx^2 + βx + γ, from zero to B ->
    I = α*B^3/3 + β*B^2/2 + γ*B
'''
# --------------------- Modules ---------------------
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from src.generate_poly_dataset import PolyDataset, ToTensor
from src import generic as gnc
from src import nn_architecture as NN
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


# --------------------- Paths ---------------------
path_to_data = os.path.join(project_dir, 'data')
path_to_models = os.path.join(project_dir, 'models')
path_to_images = os.path.join(project_dir, 'images')
data_path = os.path.join(path_to_data, 'poly_data.npy')

# --------------------- Parameters ---------------------
torch.set_default_dtype(torch.float64)
batch_size = 64
lr = 0.00001
epochs = 50
n_sample = 10000
full_data = False

# ------------------ Load data -------------------------
d = np.load(f"{path_to_data}/poly_dataset.npz", allow_pickle=True)
training_data_tensor, val_data_tensor, test_data_tensor =  ((d['train'], d['val'], d['test']))

# --------------------- Create data loaders ---------------------
train_dataloader = DataLoader(training_data_tensor,
                              batch_size=batch_size,
                              shuffle=True)
val_dataloader = DataLoader(val_data_tensor,
                            batch_size=batch_size,
                            shuffle=True)
test_dataloader = DataLoader(test_data_tensor,
                             batch_size=batch_size,
                             shuffle=True)

# --------------------- Define NN architecture ---------------------

input_size = 4
output_size = 1
network_size = [input_size] + [128] * 6 + [output_size]
model = NN.NeuralNetwork(network_size).to(dtype=torch.float32)
print(model)

# --------------------- Defining loss function and optimizer ---------------------
loss_arg = 'MSE'
metric_arg = 'RMSE'
match loss_arg:
    case 'MAE':
        loss_fn = nn.L1Loss()
    case 'MSE':
        loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print("Optimizer Details:")
print(optimizer)

# --------------------- Get model filename ---------------------
if full_data:
    data_size = None

# --------------------- Train/test the model ---------------------
if __name__ == '__main__':
    train_losses = []
    val_losses = []
    early_stopper = EarlyStopper(patience=5, min_delta=5e-3)
    for t in tqdm(range(epochs)):
        train_loss, val_loss = train(
            train_dataloader, model, loss_fn, optimizer, val_dataloader=val_dataloader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if t % 1 == 0:
            print(f"Epoch {t}\n-------------------------")
            print(
                f"Avg train loss: {train_loss:>8e}, \nAvg val loss: {val_loss:>8e} \n")
        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at:\nEpoch {t}")
            break
    print("Done!\n")

    # --------------------- Plot loss curves ---------------------
    plots = gnc.plot_loss(
        train_loss=train_losses, val_loss=val_losses)
    print(
        f": \n Min {loss_arg} for training: {(np.array(train_losses).min()):>8e}, \nMin {loss_arg} for validation: {(np.array(val_losses).min()):>8e}")

    # --------------------- Testing ---------------------
    model.eval()  # Set the model to evaluation mode
    metric = test(test_dataloader, model, metric=metric_arg)

    # predictions are unnormalized in the function
    y_pred = gnc.predict(model, X_test)
    histograms = gnc.plot_histograms(y, y_train, y_val, y_test, y_pred)

    custom_metric = 0.01
    custom_accuracy = np.mean(abs((y_pred - y_test) / y_test) < custom_metric)
    print(
        f"% of predictions inferior to {custom_metric:.0%} relative error: {custom_accuracy:.1%}")

    custom_metric = 0.05
    custom_accuracy = np.mean(abs((y_pred - y_test) / y_test) < custom_metric)
    print(
        f"% of predictions inferior to {custom_metric:.0%} relative error: {custom_accuracy:.1%}")

    custom_metric = 0.1
    custom_accuracy = np.mean(abs((y_pred - y_test) / y_test) < custom_metric)
    print(
        f"% of predictions inferior to {custom_metric:.0%} relative error: {custom_accuracy:.1%}")

    plt.show()
