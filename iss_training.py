# ------------------------- Imports -------------------
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
from src.deeponet_architecture import ISSDeepOnet
from sklearn import preprocessing as pre

#----------------------- Paths ------------------------
path_to_data = os.path.join(project_dir, 'data')
path_to_images = os.path.join(project_dir, 'images')
path_to_models = os.path.join(project_dir, 'models')
date = datetime.today().strftime('%Y%m%d')
precision = torch.float64

# ----------------------- Functions -----------------------
class ISSDataset(torch.utils.data.Dataset):
    def __init__(self, u_data, Gr_data, Gz_data):
        self.u_data = u_data
        self.Gr_data = Gr_data
        self.Gz_data = Gz_data

    def __len__(self):
        return len(self.u_data)

    def __getitem__(self, idx):
        u = self.u_data[idx]
        Gr = self.Gr_data[idx]
        Gz = self.Gz_data[idx]
        return u, Gr, Gz
    
class ComplexLoss(nn.Module):
    """Loss function for (split) complex output.

    Args:
        output_real (np.array): Prediction of real part.
        output_imag (np.array): Prediction of imaginary part.
        target (np.array): 2 column array with the labels for real and imaginary part.

    Returns:
        loss: Mean squared error of (Re(u_pred) - Re(u) + Imag(u_pred) - Imag(u)).
    """
    def __init__(self):
        super(ComplexLoss, self).__init__()

    def forward(self, output_real, output_imag, target):
        target_real = target[:,:, 0]
        target_imag = target[:,:, -1]
        loss = torch.mean(((output_real - target_real) + (output_imag - target_imag))**2)
        return loss
    
def load_data(data):
    convert_to_tensor = lambda x: torch.tensor(x, dtype=precision)
    vector_to_matrix = lambda x: x.reshape(-1,1) if type(x) == np.ndarray and x.ndim == 1 else x
    return map(convert_to_tensor, list(map(vector_to_matrix, data)))

def train_step(model, data):
    u, y, Gr, Gz = data
    model.train()
    optimizer.zero_grad()

    (Gr_real_pred, Gr_imag_pred, Gz_real_pred, Gz_imag_pred) = model(u, y)
    loss_r = loss_fn(Gr_real_pred, Gr_imag_pred, Gr)
    loss_z = loss_fn(Gz_real_pred, Gz_imag_pred, Gz)
    
    loss_r.backward(retain_graph=True)
    loss_z.backward()
    optimizer.step()

    G_pred = (Gr_real_pred, Gr_imag_pred, Gz_real_pred, Gz_imag_pred)
    loss = (loss_r, loss_z)
    return loss, G_pred

def test_step(model, data):
    u, y, Gr, Gz  = data
    model.eval()

    (Gr_real_pred, Gr_imag_pred, Gz_real_pred, Gz_imag_pred) = model(u, y)
    loss_r = loss_fn(Gr_real_pred, Gr_imag_pred, Gr)
    loss_z = loss_fn(Gz_real_pred, Gz_imag_pred, Gz)

    G_pred = (Gr_real_pred, Gr_imag_pred, Gz_real_pred, Gz_imag_pred)
    loss = (loss_r, loss_z)
    return loss, G_pred

# ---------------- Load training data -------------------
d = np.load(f"{path_to_data}/iss_train_processed.npz", allow_pickle=True)
u_train, y_train, G_train_r, G_train_z =  load_data((d['X_branch'], d['X_trunk'], d['y_r'], d['y_z']))

d = np.load(f"{path_to_data}/iss_test_processed.npz", allow_pickle=True)
u_test, y_test, G_test_r, G_test_z =  load_data((d['X_branch'], d['X_trunk'], d['y_r'], d['y_z']))

# ------------------- Normalizing -------------------------
u_train, u_test = pre.MinMaxScaler().fit_transform(u_train), pre.MinMaxScaler().fit_transform(u_test)

# ------------------------ Defining model -------------------
u_dim = u_train.shape[-1]           # Input dimension for branch net -> (mx4)
y_dim = y_train.shape[-1]                           # Input dimension for trunk net -> (qx2)
p = 40                              # Output dimension for branch and trunk net -> p
layers_f = [u_dim] + [40]*3 + [2*p]   # Branch real net MLP
layers_f = [u_dim] + [40]*3 + [2*p]   # Branch imag net MLP
layers_y = [y_dim] + [40]*3 + [p]   # trunk net MLP

model = ISSDeepOnet(layers_f, layers_y).to(precision)

# --------------- Loss function and optimizer ----------
loss_fn = ComplexLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

# ---------------- Training ----------------------
epochs = 5000
batch_size = 100

train_dataset = ISSDataset(u_train, G_train_r, G_train_z)
test_dataset = ISSDataset(u_test, G_test_r, G_test_z)

train_loss_r_list = []
test_loss_r_list = []
train_loss_z_list = []
test_loss_z_list = []

train_err_r_real_list = []
train_err_r_imag_list = []
test_err_r_real_list = []
test_err_r_imag_list = []

train_err_z_real_list = []
train_err_z_imag_list = []
test_err_z_real_list = []
test_err_z_imag_list = []

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
        u_batch_train, Gr_batch_train, Gz_batch_train = batch
        train_datapoint = u_batch_train, y_train, Gr_batch_train, Gz_batch_train
        epoch_train_loss, G_train_pred = train_step(model, train_datapoint)
    for batch in test_dataloader:
        u_batch_test, Gr_batch_test, Gz_batch_test = batch
        test_datapoint = u_batch_test, y_train, Gr_batch_test, Gz_batch_test
        epoch_test_loss, G_test_pred = test_step(model, test_datapoint)

    Gr_train_real_pred, Gr_train_imag_pred, Gz_train_real_pred, Gz_train_imag_pred = G_train_pred
    Gr_test_real_pred, Gr_test_imag_pred, Gz_test_real_pred, Gz_test_imag_pred = G_test_pred

    train_loss_r_list.append(epoch_train_loss[0])
    test_loss_r_list.append(epoch_test_loss[0])
    train_loss_z_list.append(epoch_train_loss[1])
    test_loss_z_list.append(epoch_test_loss[1])
    # lr.append(scheduler.get_last_lr())

    if i  == epochs:
        print(f"Iteration: {i} Train Loss:{epoch_train_loss}, Test Loss:{epoch_test_loss}")
    
    Gr_batch_real_train = Gr_batch_train[:,:,0]
    Gr_batch_real_test = Gr_batch_test[:,:,0]
    Gr_batch_imag_train = Gr_batch_train[:,:,1]
    Gr_batch_imag_test = Gr_batch_test[:,:,1]
    Gz_batch_real_train = Gz_batch_train[:,:,0]
    Gz_batch_real_test = Gz_batch_test[:,:,0]
    Gz_batch_imag_train = Gz_batch_train[:,:,1]
    Gz_batch_imag_test = Gz_batch_test[:,:,1]
    with torch.no_grad():
        err_r_real_train = torch.linalg.vector_norm(Gr_train_real_pred - Gr_batch_real_train) / torch.linalg.vector_norm(Gr_batch_real_train)
        err_r_real_test = torch.linalg.vector_norm(Gr_test_real_pred - Gr_batch_real_test) / torch.linalg.vector_norm(Gr_batch_real_test)

        train_err_r_real_list.append(err_r_real_train)
        test_err_r_real_list.append(err_r_real_test)

        err_r_imag_train = torch.linalg.vector_norm(Gr_train_imag_pred - Gr_batch_imag_train) / torch.linalg.vector_norm(Gr_batch_imag_train)
        err_r_imag_test = torch.linalg.vector_norm(Gr_test_imag_pred - Gr_batch_imag_test) / torch.linalg.vector_norm(Gr_batch_imag_test)

        train_err_r_imag_list.append(err_r_imag_train)
        test_err_r_imag_list.append(err_r_imag_test)

        err_z_real_train = torch.linalg.vector_norm(Gz_train_real_pred - Gz_batch_real_train) / torch.linalg.vector_norm(Gz_batch_real_train)
        err_z_real_test = torch.linalg.vector_norm(Gz_test_real_pred - Gz_batch_real_test) / torch.linalg.vector_norm(Gz_batch_real_test)

        train_err_z_real_list.append(err_z_real_train)
        test_err_z_real_list.append(err_z_real_test)

        err_z_imag_train = torch.linalg.vector_norm(Gz_train_imag_pred - Gz_batch_imag_train) / torch.linalg.vector_norm(Gz_batch_imag_train)
        err_z_imag_test = torch.linalg.vector_norm(Gz_test_imag_pred - Gz_batch_imag_test) / torch.linalg.vector_norm(Gz_batch_imag_test)

        train_err_z_imag_list.append(err_z_imag_train)
        test_err_z_imag_list.append(err_z_imag_test)
    # if i > 0 and i % (epochs//10) == 0:
    #     scheduler.step()

print(f"Train Loss: {epoch_train_loss:}, Test Loss: {epoch_test_loss}")
# print(f"Train error: {err_train:.3%}, Test error: {err_test:.3%}")

# ------------- Plots -----------------------------
x = range(epochs)

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))

# Loss for u_r
ax[0][0].plot(x, [i.item() for i in train_loss_r_list], label='train_r')
ax[0][0].plot(x, [i.item() for i in test_loss_r_list], label='test_r', linewidth=0.8)
ax[0][0].set_xlabel('epoch')
ax[0][0].set_ylabel('MSE')
# ax[0][0].set_yscale('log')
ax[0][0].set_title(r'$u_{r}$ Loss')
ax[0][0].legend()

# Loss for u_z
ax[0][1].plot(x, [i.item() for i in train_loss_z_list], label='train_z')
ax[0][1].plot(x, [i.item() for i in test_loss_z_list], label='test_z', linewidth=0.8)
ax[0][1].set_xlabel('epoch')
ax[0][1].set_ylabel('MSE')
# ax[0][1].set_yscale('log')
ax[0][1].set_title(r'$u_{z}$ Loss')
ax[0][1].legend()

# Error for Real Part of u_r
ax[1][0].plot(x, [err.item() * 100 for err in train_err_r_real_list], label='train_Re(ur)')
ax[1][0].plot(x, [err.item() * 100 for err in test_err_r_real_list], label='test_Re(ur)', linewidth=0.8)
ax_10_sec = ax[1][0].twinx()
ax[1][0].set_xlabel('epoch')
ax[1][0].set_ylabel(r"$L_2$ error [%]")
# ax[1][0].set_yscale('log')
ax[1][0].set_title(r'$Re(u_{r})$ error')
ax[1][0].legend()
# ax_1_sec.plot(x, lr, "k--", label='lr', linewidth=0.5)
# ax_1_sec.set_ylabel(r"Learning rate")
# ax_1_sec.set_yscale('log')
# ax_10_sec.legend()

# Error for Imaginary Part of u_r
ax[1][1].plot(x, [err.item() * 100 for err in train_err_r_imag_list], label='train_Im(ur)')
ax[1][1].plot(x, [err.item() * 100 for err in test_err_r_imag_list], label='test_Im(ur)', linewidth=0.8)
ax[1][1].set_xlabel('epoch')
ax[1][1].set_ylabel(r"$L_2$ error [%]")
# ax[1][1].set_yscale('log')
ax[1][1].set_title(r'$Im(u_{r})$ error')
ax[1][1].legend()

# Error for Real Part of u_z
ax[2][0].plot(x, [err.item() * 100 for err in train_err_z_real_list], label='train_Re(uz)')
ax[2][0].plot(x, [err.item() * 100 for err in test_err_z_real_list], label='test_Re(uz)', linewidth=0.8)
ax[2][0].set_xlabel('epoch')
ax[2][0].set_ylabel(r"$L_2$ error [%]")
# ax[2][0].set_yscale('log')
ax[2][0].set_title(r'$Re(u_{z})$ Error')
ax[2][0].legend()

# Error for Imaginary Part of u_z
ax[2][1].plot(x, [err.item() * 100 for err in train_err_z_imag_list], label='train_Im(uz)')
ax[2][1].plot(x, [err.item() * 100 for err in test_err_z_imag_list], label='test_Im(uz)', linewidth=0.8)
ax[2][1].set_xlabel('epoch')
ax[2][1].set_ylabel(r"$L_2$ error [%]")
# ax[2][1].set_yscale('log')
ax[2][1].set_title(r'$Im(u_{z})$ Error')
ax[2][1].legend()

fig.tight_layout()

plt.show()

fig_name = f"iss_deeponet_accuracy_plots_{date}.png"
image_path = os.path.join(path_to_images, fig_name)
fig.savefig(image_path)

# ------------ Saving model ----------------
model_name = f"iss_model_{date}.pth"
torch.save(model, os.path.join(path_to_models, model_name))