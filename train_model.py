import os
import numpy as np
import time
import yaml
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datetime import datetime
from modules.preprocessing import preprocessing
from modules.plotting import plot_training

class MLP(nn.Module):

    def __init__(self, layers, activation):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(
                nn.Linear(
                    layers[i],
                    layers[i+1],
                )
            )
        self.activation = activation

    def forward(self, inputs):
        out = inputs
        for i in range(len(self.linears) - 1):
            out = self.linears[i](out)
            out = self.activation(out)
        return self.linears[-1](out)
    
def train_step():
    optimizer.zero_grad()
    
    out_B = branch(u_train)
    out_B_real = out_B[:,:G_dim]
    out_B_imag = out_B[:,G_dim:]
    out_T = trunk(xt)   
    g_u_pred_real = torch.matmul(out_B_real, torch.transpose(out_T, 0, 1))
    g_u_pred_imag = torch.matmul(out_B_imag, torch.transpose(out_T, 0, 1))

    loss_real = torch.mean((g_u_pred_real - g_u_real_train) ** 2)
    loss_imag = torch.mean((g_u_pred_imag - g_u_imag_train) ** 2)
    loss = loss_real + loss_imag
    loss.backward()
    optimizer.step()

    return loss.item(), g_u_pred_real, g_u_pred_imag

def test_step():
    with torch.no_grad():  
        optimizer.zero_grad()
        out_B = branch(u_test)
        out_B_real = out_B[:,:G_dim]
        out_B_imag = out_B[:,G_dim:]
        out_T = trunk(xt)   
        g_u_pred_real = torch.matmul(out_B_real, torch.transpose(out_T, 0, 1))
        g_u_pred_imag = torch.matmul(out_B_imag, torch.transpose(out_T, 0, 1))
    return g_u_pred_real, g_u_pred_imag

# ------------ Parameters --------------
with open('params_model.yaml') as file:
    p = yaml.safe_load(file)

date = datetime.today().strftime('%Y%m%d')
precision = eval(p['PRECISION'])
device = torch.device(p["DEVICE"])
print("Using device:", device)

# -------------- Data processing -------------------
# data = sio.loadmat(p["DATAFILE"])
data = np.load(p['DATAFILE'])
omega = np.squeeze(data['freqs'])        # Shape: (num_freqs,)
r = np.squeeze(data['r_field'])          # Shape: (n,)
z = np.squeeze(data['z_field'])          # Shape: (n,)
wd = data['wd'].transpose(2,0,1)                           # Shape: (freqs, r, z)

u = omega.reshape(-1,1)
R,Z = np.meshgrid(r,z)
xt = np.column_stack((R.flatten(), Z.flatten()))
g_u = wd.reshape(len(u),-1)

processed_data = preprocessing(u, g_u, train_perc=p["TRAIN_PERC"])

u_train, g_u_real_train, g_u_imag_train, u_test, g_u_real_test, g_u_imag_test = \
    processed_data['u_train'], \
    processed_data['g_u_real_train'],\
    processed_data['g_u_imag_train'], \
    processed_data['u_test'], \
    processed_data['g_u_real_test'], \
    processed_data['g_u_imag_test']

mu_u = np.mean(u_train, axis=0)
sd_u = np.std(u_train, axis=0)
mu_xt = np.mean(xt, axis=0)
sd_xt = np.std(xt, axis=0)

u_train = (u_train - mu_u) / sd_u
u_test = (u_test - mu_u) / sd_u

xt = (xt - mu_xt)/sd_xt

print(f"Branch normalization: mu={mu_u.item():.2f}, sd={sd_u.item():.2f}")
print(f"Branch train/test shapes: {u_train.shape}, {u_test.shape}")
print(xt.shape)
print(g_u_real_train.shape, g_u_imag_train.shape)

u_train = torch.tensor(u_train, dtype=precision, device=device)
u_test = torch.tensor(u_test, dtype=precision, device=device)
xt = torch.tensor(xt, dtype=precision, device=device)
g_u_real_train = torch.tensor(g_u_real_train, dtype=precision, device=device)
g_u_imag_train = torch.tensor(g_u_imag_train, dtype=precision, device=device)
g_u_real_test = torch.tensor(g_u_real_test, dtype=precision, device=device)
g_u_imag_test = torch.tensor(g_u_imag_test, dtype=precision, device=device)

# ---------- Model definition -----------
u_dim = p["BRANCH_INPUT_SIZE"]
G_dim = p["BASIS_FUNCTIONS"]
x_dim = p["TRUNK_INPUT_SIZE"]

layers_B = [u_dim] + [100] * 5 + [G_dim*2]
layers_T = [x_dim] + [100] * 5 + [G_dim]

branch = MLP(layers=layers_B, activation=nn.ReLU()).to(device, dtype=precision)
trunk = MLP(layers=layers_T, activation=nn.ReLU()).to(device, dtype=precision)

optimizer = torch.optim.Adam(list(branch.parameters()) + list(trunk.parameters()), lr=p["LEARNING_RATE"])

num_epochs = p['N_EPOCHS']
niter_per_epoch = p['ITERATIONS_PER_EPOCHS']  
t0 = time.time()

# ------------------- Training loop -----------------
train_loss_list = []
train_err_real_list = []
train_err_imag_list = []
test_err_real_list = []
test_err_imag_list = []

for epoch in range(num_epochs):
    epoch_loss = 0

    for _ in range(niter_per_epoch):
        loss, g_u_pred_real_train, g_u_pred_imag_train = train_step()
        epoch_loss += loss

    avg_epoch_loss = epoch_loss / niter_per_epoch

    with torch.no_grad():
        g_u_pred_real_test, g_u_pred_imag_test = test_step()  

        train_err_real = torch.linalg.vector_norm(g_u_pred_real_train - g_u_real_train) / torch.linalg.vector_norm(g_u_real_train)
        train_err_imag = torch.linalg.vector_norm(g_u_pred_imag_train - g_u_imag_train) / torch.linalg.vector_norm(g_u_imag_train)

        train_err_real_list.append(train_err_real)
        train_err_imag_list.append(train_err_imag)

        test_err_real = torch.linalg.vector_norm(g_u_pred_real_test - g_u_real_test) / torch.linalg.vector_norm(g_u_real_test)
        test_err_imag = torch.linalg.vector_norm(g_u_pred_imag_test - g_u_imag_test) / torch.linalg.vector_norm(g_u_imag_test)

        test_err_real_list.append(test_err_real)
        test_err_imag_list.append(test_err_imag)

    if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_epoch_loss}")

    train_loss_list.append(avg_epoch_loss)

t1 = time.time()
print(f"Training completed in {t1 - t0:.2f} seconds")

# ---------- Plots ----------------
x = range(num_epochs)

# plot_training(x, train_loss_list, train_err_real_list, train_err_imag_list, test_err_real_list, test_err_imag_list)

# ----------- Save output ------------


filename = p["TEST_PREDS_DATA_FILE"]
try:
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
except FileExistsError as e:
    print('Rewriting previous data file...')

np.savez(filename, u=u_test, xt=xt, real=g_u_pred_real_test, imag=g_u_pred_imag_test, mu_u=mu_u, sd_u=sd_u, mu_xt=mu_xt, sd_xt=sd_xt)
