import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from src.deeponet_architecture import FNNDeepOnet

def load_data(data):
    convert_to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    vector_to_matrix = lambda x: x.reshape(-1,1) if type(x) == np.ndarray and x.ndim == 1 else x
    
    return map(convert_to_tensor, list(map(vector_to_matrix, data)))

def train_step(model, data):
    u, y, G = data

    optimizer.zero_grad()

    G_pred = model(u, y)
    loss = loss_fn(G_pred, G)
    
    loss.backward()
    optimizer.step()

    return loss, G_pred

def test_step(model, data):
    u, y, G = data

    optimizer.zero_grad()

    G_pred = model(u, y)
    loss = loss_fn(G_pred, G)

    return loss, G_pred

path_to_data = os.path.join(project_dir, 'data')

# ---------------- Load training data -------------------
d = np.load(f"{path_to_data}/antiderivative_train.npz", allow_pickle=True)
u_train, y_train, G_u_y_train =  load_data((d['X_branch'], d['X_trunk'], d['y']))

d = np.load(f"{path_to_data}/antiderivative_train.npz", allow_pickle=True)
u_test, y_test, G_u_y_test =  load_data((d['X_branch'], d['X_trunk'], d['y']))

# ---------------- Defining model -------------------
u_dim = u_train.shape[0]           # Input dimension for branch net -> m
p = 50                              # Output dimension for branch and trunk net -> p
layers_f = [u_dim] + [40]*2 + [p]   # Branch net MLP
y_dim = 1                           # Input dimension for trunk net -> q
layers_y = [y_dim] + [40]*2 + [p]   # Branch net MLP

model = FNNDeepOnet(layers_f, layers_y)
print(model(u_train,y_train))

# --------------- Loss function and optimizer ----------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------- Training ----------------------
batch_size = 32
epochs = 10000

train_set = (u_train, y_train, G_u_y_train)
test_set = (u_test, y_test, G_u_y_test)

train_err_list = []
train_loss_list = []
test_err_list = []
test_loss_list = []

for i in tqdm(range(epochs)):
    epoch_train_loss, G_train_pred = train_step(model, train_set, u_dim, y_dim)
    epoch_test_loss, G_test_pred = test_step(model, test_set, u_dim, y_dim)

    train_loss_list.append(epoch_train_loss)
    test_loss_list.append(epoch_test_loss)

    if i % 100 == 0:
        print(f"Iteration: {i} Train Loss:{epoch_train_loss}, Test Loss:{epoch_test_loss}")
    with torch.no_grad():
        print(G_train_pred.shape, G_u_y_train.shape)
        err_train = torch.linalg.vector_norm(G_train_pred - G_u_y_train) / torch.linalg.vector_norm(G_u_y_train)
        err_test = torch.linalg.vector_norm(G_test_pred - G_u_y_test) / torch.linalg.vector_norm(G_u_y_test)
        train_err_list.append(err_train)
        test_err_list.append(err_test)