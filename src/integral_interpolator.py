# --------------------- Modules ---------------------
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import nn
from src import generic as gnc
from src import nn_architecture as NN
import numpy as np
import torch

precision = torch.float32

# --------------------- Paths ---------------------
path_to_data = os.path.join(project_dir, 'data')
path_to_models = os.path.join(project_dir, 'models')
path_to_images = os.path.join(project_dir, 'images')
data_path = os.path.join(path_to_data, 'poly_data.npy')

class PolyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        features_tensor, label_tensor = list(map(lambda x: torch.tensor(x, dtype=precision), [features, label]))
        return features_tensor, label_tensor

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

    return loss, y_pred

def test(data, model):
    X, y = data
    model.eval()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    return loss, y_pred

# --------------------- Parameters ---------------------
batch_size = 128
epochs = 500

# ------------------ Load data -------------------------
d = np.load(f"{path_to_data}/mlp_dataset_train.npz", allow_pickle=True)
X_train, y_train =  ((d['X'], d['y']))

d = np.load(f"{path_to_data}/mlp_dataset_val.npz", allow_pickle=True)
X_val, y_val =  ((d['X'], d['y']))

d = np.load(f"{path_to_data}/mlp_dataset_test.npz", allow_pickle=True)
X_test, y_test =  ((d['X'], d['y']))


train_dataset = PolyDataset(X_train, y_train)
val_dataset = PolyDataset(X_val, y_val)
test_dataset = PolyDataset(X_test, y_test)

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
network_size = [input_size] + [256] * 6 + [output_size]
model = NN.NeuralNetwork(network_size).to(dtype=precision)

# --------------------- Defining loss function and optimizer ---------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------- Train/test the model ---------------------
if __name__ == '__main__':
    train_losses = []
    val_losses = []
    # early_stopper = EarlyStopper(patience=5, min_delta=5e-3)
    for t in tqdm(range(epochs), colour='BLUE'):
        for batch in train_dataloader:
            X_batch_train, y_batch_train = batch
            train_point = X_batch_train, y_batch_train
            train_loss, y_train = train(train_point, model)
        for batch in val_dataloader:
            X_batch_val, y_batch_val = batch
            val_point = X_batch_val, y_batch_val
            val_loss, y_val = test(val_point, model)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if t % 1 == 0:
            print(f"Epoch {t}\n-------------------------")
            print(
                f"Avg train loss: {train_loss:>8e}, \nAvg val loss: {val_loss:>8e} \n")
        # if early_stopper.early_stop(val_loss):
        #     print(f"Early stopping at:\nEpoch {t}")
        #     break
    print("Done!\n")

