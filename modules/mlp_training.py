# --------------------- Modules ---------------------
import os
import sys
from datetime import datetime
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import nn
from src import mlp_architecture as NN
import numpy as np
import torch

precision = torch.float32

# --------------------- Paths ---------------------
path_to_data = os.path.join(project_dir, 'data')
path_to_models = os.path.join(project_dir, 'models')
path_to_images = os.path.join(project_dir, 'images')
date = datetime.today().strftime('%Y%m%d')

class PolyDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=precision)
        self.labels = torch.tensor(y, dtype=precision)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        return features, label

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

def train(data, model):
    X, y = data
    model.train()
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    return loss.detach(), y_pred

def val(data, model):
    X, y = data
    model.eval()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    return loss.detach(), y_pred

def test(data, model):
    X, y = data
    model.eval()
    y_pred = model(X)
    error = torch.linalg.vector_norm(y_pred - y) / torch.linalg.vector_norm(y)

    return error.detach()

def compute_integral(coefs, a=0.1):
    alpha, beta, gamma, B = coefs[:,0], coefs[:, 1], coefs[:, 2], coefs[:, 3]
    integrals = (alpha / 3) * (B**3 - a**3) + (beta / 2) * (B**2 - a**2) + \
            gamma * (B - a)
    return integrals

# --------------------- Parameters ---------------------
batch_size = 128
epochs = 100

# ------------------ Load data -------------------------
d = np.load(f"{path_to_data}/mlp_dataset_train.npz", allow_pickle=True) # If this doesn't run, generate the dataset by running the mlp_data.py file/
X_train, y_train =  ((d['X'], d['y']))

d = np.load(f"{path_to_data}/mlp_dataset_val.npz", allow_pickle=True)
X_val, y_val =  ((d['X'], d['y']))

d = np.load(f"{path_to_data}/mlp_dataset_test.npz", allow_pickle=True)
X_test, y_test =  ((d['X'], d['y']))


train_dataset = PolyDataset(X_train, y_train)
val_dataset = PolyDataset(X_val, y_val)
test_dataset = PolyDataset(X_test, y_test)

X_train, X_val, X_test = list(map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_val, X_test]))
y_train, y_val, y_test = list(map(lambda x: torch.tensor(x, dtype=torch.float32), [y_train, y_val, y_test]))

# --------------------- Create data loaders ---------------------
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

# --------------------- Define NN architecture ---------------------
input_size = 4
output_size = 1
network_size = [input_size] + [40] * 3 + [output_size]
model = NN.NeuralNetwork(network_size).to(dtype=precision)

# --------------------- Defining loss function and optimizer ---------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# --------------------- Train/test the model ---------------------
if __name__ == '__main__':
    train_losses = []
    val_losses = []
    test_error = []
    # early_stopper = EarlyStopper(patience=5, min_delta=5e-3)
    for t in tqdm(range(epochs), colour='BLUE'):
        for batch in train_dataloader:
            train_loss, y_pred_train = train(batch, model)
        for batch in val_dataloader:
            val_loss, y_pred_val = val(batch, model)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if t % 1 == 0:
            print(f"Epoch {t}\n-------------------------")
            print(
                f"Avg train loss: {train_loss:>8e}, \nAvg val loss: {val_loss:>8e} \n")
            
        with torch.no_grad():
            y_pred_test = model(X_test)
            error = torch.linalg.vector_norm(y_pred_test - y_test) / torch.linalg.vector_norm(y_test)

        test_error.append(error.item())

        # if early_stopper.early_stop(val_loss):
        #     print(f"Early stopping at:\nEpoch {t}")
        #     break
    print("Done!\n")
    print(f"Final loss: {train_loss:.5E}, Final error: {error:.3%}")


fig, ax = plt.subplots(ncols=2, figsize=(10,5))

x = range(epochs)

ax[0].plot(x, train_losses, label='train')
ax[0].plot(x, val_losses, label='val')
ax[0].set_yscale('log')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('MSE')
ax[0].legend()

ax[1].plot(x, [10*i for i in test_error], '-g',  label='test error')
ax[1].set_yscale('log')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel(r"$L_2$ error [%]")
ax[1].legend()

plt.show()


fig_name = f"MLP_accuracy_plots_{date}.png"
image_path = os.path.join(path_to_images, fig_name)
fig.savefig(image_path)

# ------------ Saving model ----------------
model_name = f"MLP_model_{date}.pth"
torch.save(model, os.path.join(path_to_models, model_name))
