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
from src import model_training as mt
from src import nn_architecture as NN
import numpy as np
import torch

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

# --------------------- Get and normalize dataset ---------------------
data = np.load(data_path)  # Load polynomials numpy array
data = data.astype(np.float64)
if not full_data:
    data = data[:n_sample]

X, y = data[:, :-1], data[:, -1]
data_norm = np.zeros((len(data), 5))
data_norm[:, :-1] = gnc.normalize(X)
data_norm[:, -1] = y
# --------------------- Format as torch and split train-test set ---------------------
training_data_params, val_data_params, test_data_params = gnc.split_dataset(data, seed=42)
mean_for_normalization, std_for_normalization = np.mean(training_data_params, axis=0)[:-1], np.std(training_data_params, axis=0)[:-1]

np.savez(f'{path_to_models}/normalization_params.npz', mean=mean_for_normalization, std=std_for_normalization)

training_data, val_data, test_data = gnc.split_dataset(data_norm, seed=42)

for i in [training_data, val_data, test_data]:
    i = i[:, :-1], i[:, -1]  # format for dataset

data_norm = data_norm[:, :-1], data_norm[:, -1]  # format for dataset
data_size = len(data_norm[0]) if not full_data else None

X_train, y_train = training_data[:, :-1], training_data[:, -1]
X_val, y_val = val_data[:, :-1], val_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]
X_test_mean, X_test_std = np.mean(X_test, axis=0), np.std(X_test, axis=0)
y_test_mean, y_test_std = np.mean(y_test, axis=0), np.std(y_test, axis=0)

training_data_tensor = PolyDataset(
    training_data, transform=ToTensor())
val_data_tensor = PolyDataset(val_data, transform=ToTensor())
test_data_tensor = PolyDataset(test_data, transform=ToTensor())
X, y = PolyDataset(data_norm, transform=ToTensor()).features, PolyDataset(
    data_norm, transform=ToTensor()).labels
print(X.shape, y.shape)

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
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(f"Using {device} device")
model = NN.NeuralNetwork().to(device, dtype=torch.float64)
nodes_config = NN.nodes_config
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
model_name = gnc.get_model_name(
    nodes_config, batch_size, lr, epochs, data_size)
plot_path = os.path.join(path_to_images, model_name)

# --------------------- Train/test the model ---------------------
if __name__ == '__main__':
    train_losses = []
    val_losses = []
    early_stopper = mt.EarlyStopper(patience=5, min_delta=5e-3)
    for t in tqdm(range(epochs)):
        train_loss, val_loss = mt.train(
            train_dataloader, model, loss_fn, optimizer, val_dataloader=val_dataloader, device=device)
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
        train_loss=train_losses, val_loss=val_losses, model_name=model_name)
    print(
        f": \n Min {loss_arg} for training: {(np.array(train_losses).min()):>8e}, \nMin {loss_arg} for validation: {(np.array(val_losses).min()):>8e}")

    # --------------------- Save model ---------------------

    # --------------------- Testing ---------------------
    model.eval()  # Set the model to evaluation mode
    metric = mt.test(test_dataloader, model, metric=metric_arg, device=device)

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
