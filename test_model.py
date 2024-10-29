import os
import numpy as np
import time
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datetime import datetime
from modules.preprocessing import preprocessing
from modules.plotting import plot_label_axis, plot_label_contours

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
    

data = np.load('new_test.npz', allow_pickle=True)

omega = np.squeeze(data['freqs'])        # Shape: (num_freqs,)
r = np.squeeze(data['r_field'])          # Shape: (n,)
z = np.squeeze(data['z_field'])          # Shape: (n,)
wd = data['wd'].transpose(0,2,1)                          # Shape: (freqs, r, z)

mu = 1.97
sd = 0.06

u = omega.reshape(-1,1)*sd + mu
R,Z = np.meshgrid(r,z)
xt = np.column_stack((R.flatten(), Z.flatten()))

u, xt = [torch.tensor(i, dtype=torch.float64) for i in [u, xt]]

branch = MLP([1] + [100] * 3 + [20*2], nn.ReLU()).to(torch.float64)
branch.load_state_dict(torch.load('./models/branch_20241023.pth', weights_only=True))

trunk = MLP([2] + [100] * 3 + [20], nn.ReLU()).to(torch.float64)
trunk.load_state_dict(torch.load('./models/trunk_20241023.pth', weights_only=True))

out_b = branch(u)

out_B_real = out_b[:,:20]
out_B_imag = out_b[:,20:]

out_t = trunk(xt)

out_real = torch.matmul(out_B_real, torch.transpose(out_t, 0, 1))
out_imag = torch.matmul(out_B_imag, torch.transpose(out_t, 0, 1))

index = 0

freq = omega[index]
g_u = out_real + out_imag*1j
g_u = g_u.reshape(-1, len(r), len(z))
g_u = g_u[index].detach().numpy()


print(np.flip(g_u[1:,:], axis=0))

fig = plot_label_contours(r, z, g_u, freq, full=True, non_dim_plot=True)
plt.show()

fig = plot_label_axis(r, z, g_u, freq, axis='r', non_dim_plot=True)

plt.show()