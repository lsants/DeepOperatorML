import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import yaml
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datetime import datetime
from torch import autograd
from modules.preprocessing import preprocessing
from modules.plotting import contour_plot

with open('params_model.yaml') as file:
    p = yaml.safe_load(file)

date = datetime.today().strftime('%Y%m%d')
precision = eval(p['PRECISION'])

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
    
# -------------- Data preprocessing -------------------
data = sio.loadmat(p["DATAFILE"])
omega = np.squeeze(data['freqs'])        # Shape: (num_freqs,)
r = np.squeeze(data['r_campo'])          # Shape: (n,)
z = np.squeeze(data['z_campo'])          # Shape: (n,)
wd = data['wd'].transpose(2,0,1)                           # Shape: (freqs, r, z)

u = omega.reshape(-1,1)
R,Z = np.meshgrid(r,z)
xt = np.column_stack((R.flatten(), Z.flatten()))

xt_mean = np.mean(xt, axis=0)
xt_std = np.std(xt, axis=0)

xt = (xt - xt_mean) / xt_std

g_u = wd.reshape(len(u),-1)

processed_data = preprocessing(u, g_u, train_perc=0.8)

u_train, g_u_real_train, g_u_imag_train, u_test, g_u_real_test, g_u_imag_test = \
    processed_data['u_train'], \
    processed_data['g_u_real_train'],\
    processed_data['g_u_imag_train'], \
    processed_data['u_test'], \
    processed_data['g_u_real_test'], \
    processed_data['g_u_imag_test']

print(u_train.shape, u_test.shape)
print(xt.shape)
print(g_u_real_train.shape, g_u_imag_train.shape)


mu = np.mean(u_train, axis=0)
sd = np.std(u_train, axis=0)
u_train = (u_train - mu) / sd
u_test = (u_test - mu) / sd

device = torch.device("cpu")
print("Using device:", device)

u_train = torch.tensor(u_train, dtype=precision).to(device)
xt = torch.tensor(xt, dtype=precision).to(device)
g_u_real_train = torch.tensor(g_u_real_train, dtype=precision).to(device)
g_u_imag_train = torch.tensor(g_u_imag_train, dtype=precision).to(device)
u_train = torch.tensor(u_train, dtype=precision).to(device)
u_test = torch.tensor(u_test, dtype=precision).to(device)
g_u_real_test = torch.tensor(g_u_real_test, dtype=precision).to(device)
g_u_imag_test = torch.tensor(g_u_imag_test, dtype=precision).to(device)

u_dim = 1
G_dim = 30
layers_B = [u_dim] + [100] * 5 + [G_dim*2]
x_dim = 2
layers_T = [x_dim] + [100] * 5 + [G_dim]

branch = MLP(layers=layers_B, activation=torch.tanh).to(device, dtype=precision)
trunk = MLP(layers=layers_T, activation=torch.tanh).to(device, dtype=precision)

optimizer = torch.optim.Adam(list(branch.parameters()) + list(trunk.parameters()), lr=1e-2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 20000, 30000], gamma=0.5)


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
    return loss.item() 


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


num_epochs = p['N_EPOCHS']
niter_per_epoch = 100  
t0 = time.time()

train_loss_list = []

train_err_real_list = []
train_err_imag_list = []
test_err_real_list = []
test_err_imag_list = []


for epoch in range(num_epochs):
    epoch_loss = 0

    for _ in range(niter_per_epoch):
        loss = train_step()
        epoch_loss += loss

    avg_epoch_loss = epoch_loss / niter_per_epoch

    with torch.no_grad():
        out_B = branch(u_train)
        out_T = trunk(xt)   
        g_u_pred_real_train = torch.matmul(branch(u_train)[:,:G_dim], torch.transpose(trunk(xt), 0, 1))
        g_u_pred_imag_train = torch.matmul(branch(u_train)[:,G_dim:], torch.transpose(trunk(xt), 0, 1))
        train_loss = torch.mean(((g_u_pred_real_train - g_u_real_train) + (g_u_pred_imag_train - g_u_imag_train)) ** 2).item()
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
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")

    train_loss_list.append(epoch_loss)


t1 = time.time()
print(f"Training completed in {t1 - t0:.2f} seconds")

x = range(num_epochs)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,9))

ax[0][0].plot(x, [i for i in train_loss_list], label='train_z')
ax[0][0].set_xlabel('epoch')
ax[0][0].set_ylabel('MSE')
ax[0][0].set_yscale('log')
ax[0][0].set_title(r'$u_{z}$ Loss')
ax[0][0].legend()

ax[0][1].plot(x, [np.sqrt(i**2 + j**2) for i,j in zip(train_err_real_list, train_err_imag_list)], label='abs_train')
ax[0][1].plot(x, [np.sqrt(i**2 + j**2) for i,j in zip(test_err_real_list, test_err_imag_list)], label='abs_test')
ax[0][1].set_xlabel('epoch')
ax[0][1].set_ylabel(r'$L_2$ norm')
ax[0][1].set_yscale('log')
ax[0][1].set_title(r'Error for $|u_{z}|$')
ax[0][1].legend()

ax[1][0].plot(x, [i for i in train_err_real_list], label='real_train')
ax[1][0].plot(x, [i for i in test_err_real_list], label='real_test')
ax[1][0].set_xlabel('epoch')
ax[1][0].set_ylabel(r'$L_2$ norm')
ax[1][0].set_yscale('log')
ax[1][0].set_title(r'Error for $Re(u_{z})$')
ax[1][0].legend()

ax[1][1].plot(x, [i for i in train_err_imag_list], label='imag_train')
ax[1][1].plot(x, [i for i in test_err_imag_list], label='imag_test')
ax[1][1].set_xlabel('epoch')
ax[1][1].set_ylabel(r'$L_2$ norm')
ax[1][1].set_yscale('log')
ax[1][1].set_title(r'Error for $Im(u_{z})$')
ax[1][1].legend()

plt.grid
plt.show()

fig_name = f"iss_deeponet_accuracy_plots_{date}.png"
image_path = ''

fig.savefig(image_path)

# ------------ Saving model ----------------
torch.save(branch, p['MODEL_FOLDER'] + p['BRANCH_MODELNAME'] + date + '.pth')
torch.save(trunk, p['MODEL_FOLDER'] + p['TRUNK_MODELNAME'] + date + '.pth')

# contour_plot(u_test.flatten(), xt[:,0], xt[:,1], g_u_real_test[0,:])