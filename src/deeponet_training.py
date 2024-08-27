import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from datetime import datetime
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from src.deeponet_architecture import FNNDeepOnet

path_to_data = os.path.join(project_dir, 'data')
path_to_images = os.path.join(project_dir, 'images')
path_to_models = os.path.join(project_dir, 'models')
date = datetime.today().strftime('%Y%m%d')

precision = torch.float32

class DeepONetDataset(torch.utils.data.Dataset):
    def __init__(self, u_data, G_data):
        self.u_data = u_data
        self.G_data = G_data

    def __len__(self):
        return len(self.u_data)

    def __getitem__(self, idx):
        u = self.u_data[idx]
        G = self.G_data[idx]
        return u, G

def load_data(data):
    convert_to_tensor = lambda x: torch.tensor(x, dtype=precision)
    vector_to_matrix = lambda x: x.reshape(-1,1) if type(x) == np.ndarray and x.ndim == 1 else x
    
    return map(convert_to_tensor, list(map(vector_to_matrix, data)))

def train_step(model, data):
    u, y, G = data

    model.train()
    optimizer.zero_grad()

    G_pred = model(u, y)
    loss = loss_fn(G_pred, G)
    
    loss.backward()
    optimizer.step()

    return loss, G_pred

def test_step(model, data):
    u, y, G = data

    model.eval()

    G_pred = model(u, y)
    loss = loss_fn(G_pred, G)

    return loss, G_pred


# ---------------- Load training data -------------------
d = np.load(f"{path_to_data}/antiderivative_train.npz", allow_pickle=True)
u_train, y_train, G_train =  load_data((d['X_branch'], d['X_trunk'], d['y']))


d = np.load(f"{path_to_data}/antiderivative_test.npz", allow_pickle=True)
u_test, y_test, G_test =  load_data((d['X_branch'], d['X_trunk'], d['y']))

# ---------------- Defining model -------------------
u_dim = u_train.shape[-1]           # Input dimension for branch net -> m
y_dim = 1                           # Input dimension for trunk net -> q
p = 40                              # Output dimension for branch and trunk net -> p
layers_f = [u_dim] + [40]*3 + [p]   # Branch net MLP
layers_y = [y_dim] + [40]*3 + [p]   # trunk net MLP

model = FNNDeepOnet(layers_f, layers_y).to(precision)

# --------------- Loss function and optimizer ----------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

# ---------------- Training ----------------------
epochs = 2000
batch_size = 100

train_dataset = DeepONetDataset(u_train, G_train)
test_dataset = DeepONetDataset(u_test, G_test)

train_err_list = []
train_loss_list = []
test_err_list = []
test_loss_list = []
lr = []

train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
)

for i in tqdm(range(epochs), colour='GREEN'):
    for batch in train_dataloader:
        u_batch_train, G_batch_train = batch
        train_datapoint = u_batch_train, y_train, G_batch_train
        epoch_train_loss, G_train_pred = train_step(model, train_datapoint)
    for batch in test_dataloader:
        u_batch_test, G_batch_test = batch
        test_datapoint = u_batch_test, y_train, G_batch_test
        epoch_test_loss, G_test_pred = test_step(model, test_datapoint)

    train_loss_list.append(epoch_train_loss)
    test_loss_list.append(epoch_test_loss)
    lr.append(scheduler.get_last_lr())

    if i  == epochs:
        print(f"Iteration: {i} Train Loss:{epoch_train_loss}, Test Loss:{epoch_test_loss}")
    with torch.no_grad():
        err_train = torch.linalg.vector_norm(G_train_pred - G_batch_train) / torch.linalg.vector_norm(G_batch_train)
        err_test = torch.linalg.vector_norm(G_test_pred - G_batch_test) / torch.linalg.vector_norm(G_batch_test)
        train_err_list.append(err_train)
        test_err_list.append(err_test)

    if i > 0 and i % (epochs//10) == 0:
        scheduler.step()

print(f"Train Loss: {epoch_train_loss:}, Test Loss: {epoch_test_loss}")
print(f"Train error: {err_train:.3%}, Test error: {err_test:.3%}")

# ------------- Plots -----------------------------
x = range(epochs)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

ax[0].plot(x, [i.item() for i in train_loss_list], label='train')
ax[0].plot(x, [i.item() for i in test_loss_list], label='test', linewidth=0.5)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('MSE')
ax[0].set_yscale('log')
ax[0].legend()

ax[1].plot(x, train_err_list, label='train')
ax[1].plot(x, test_err_list, label='test', linewidth=0.5)
ax_1_sec = ax[1].twinx()
ax_1_sec.plot(x, lr, "k--", label='lr', linewidth=0.5)
ax[1].set_xlabel('epoch')
ax[1].set_ylabel(r"$L_2$ error [%]")
ax_1_sec.set_ylabel(r"Learning rate")
ax[1].set_yscale('log')
ax_1_sec.set_yscale('log')
ax[1].legend()
ax_1_sec.legend()

fig.tight_layout()

plt.show()

fig_name = f"deeponet_accuracy_plots_{date}.png"
image_path = os.path.join(path_to_images, fig_name)
fig.savefig(image_path)

# ------------ Saving model ----------------
model_name = f"deeponet_model_{date}.pth"
torch.save(model, os.path.join(path_to_models, model_name))

