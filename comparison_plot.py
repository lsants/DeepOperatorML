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

with open('params_model.yaml') as file:
    p = yaml.safe_load(file)

date = datetime.today().strftime('%Y%m%d')
precision = eval(p['PRECISION'])

plot = input()

# ----------------Get Labels -----------------
path = '/users/lsantia9/Documents/base script 160321/influencia_data.mat'
data_labels = sio.loadmat(path)

# Extract variables and remove singleton dimensions
omega = np.squeeze(data_labels['freqs'])        # Shape: (num_freqs,)
r = np.squeeze(data_labels['r_campo'])          # Shape: (n,)
z = np.squeeze(data_labels['z_campo'])          # Shape: (n,)
wd = data_labels['wd'].transpose(0,1,2)                           # Shape: (n, n, num_freqs)


# Print shapes for debugging
print('omega shape:', omega.shape)
print('r shape:', r.shape)
print('z shape:', z.shape)
print('u shape:', wd.shape)

wd = wd.reshape(len(r), len(z), len(omega))

freq_index = 4
wd = wd[:,:,freq_index]

wd_flip = np.flip(wd[1:, :], axis=0)
wd_full = np.concatenate((wd_flip, wd), axis=0)

# -------------- Get Preds -------------------
data_preds = np.load("/users/lsantia9/research/high_performance_integration/data/info/test_output.npz", allow_pickle=True)

u, xt, g_u_real, g_u_imag, mu_u, sd_u, mu_xt, sd_xt = data_preds["u"], data_preds["xt"], data_preds["real"], data_preds["imag"], data_preds['mu_u'], data_preds['sd_u'], data_preds['mu_xt'], data_preds['sd_xt']

f_pred = (u[0].flatten()*sd_u + mu_u).item()

r, z = xt[:,0][:10]*sd_xt[0] + mu_xt[0], np.unique(xt[:,1])*sd_xt[1] + mu_xt[1]

r_full = np.concatenate((-np.flip(r[1:]), r))

g_u_real, g_u_imag = g_u_real.reshape(len(r), len(z)), g_u_imag.reshape(len(r), len(z))

print(f"Test freqs: {u.flatten()*sd_u + mu_u}")
print(f"Test coords: {xt.shape}")
print(f"Test g_u real: {g_u_real.shape}")
print(f"Test g_u imag: {g_u_imag.shape}")


try:
    if plot == 'abs':
        g_u = np.sqrt(g_u_real**2 + g_u_imag**2)
        wd_plot = np.abs(wd_full)
        l = r'|$u_z$|'
    elif plot == 'real':
        wd_plot = np.real(wd_full)
        g_u = g_u_real
        l = r'Re($u_z$)'
    elif plot == 'imag':
        g_u = g_u_imag
        wd_plot = np.imag(wd_full)
        l = r'Im($u_z$)'
    else:
        raise ValueError("Invalid plot command")
except ValueError as e:
    print(e.args)

g_u_flip = np.flip(g_u[1:, :], axis=0)
g_u_full = np.concatenate((g_u_flip, g_u), axis=0)

#  --------------- Plots ----------------
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

contour_preds = ax[0].contourf(r_full,z,g_u_full.T,  cmap="viridis")
ax[0].invert_yaxis()
ax[0].set_xlabel('r')
ax[0].set_ylabel('z')
if plot == 'abs':
    ax[0].set_title(r'$|u_{z_{\mathrm{pred}}}|$' + f' at ω = {f_pred:.2f} rad/s')
elif plot == 'real':
    ax[0].set_title(r'$\Re(u_{z_{\mathrm{pred}}})$' + f' at ω = {f_pred:.2f} rad/s')
else:
    ax[0].set_title(r'$\Im(u_{z_{\mathrm{pred}}})$' + f' at ω = {f_pred:.2f} rad/s')

contour_labels = ax[1].contourf(r_full,z,wd_plot.T,  cmap="viridis")
ax[1].invert_yaxis()
ax[1].set_xlabel('r')
ax[1].set_ylabel('z')
if plot == 'abs':
    ax[1].set_title(r'$|u_{z_{\mathrm{label}}}|$' + f' at ω = {omega[freq_index]:.2f} rad/s')
elif plot == 'real':
    ax[1].set_title(r'$\Re(u_{z_{\mathrm{label}}})$' + f' at ω = {omega[freq_index]:.2f} rad/s')
else:
    ax[1].set_title(r'$\Im(u_{z_{\mathrm{label}}})$' + f' at ω = {omega[freq_index]:.2f} rad/s')

cbar_labels = fig.colorbar(contour_labels, label=l)
cbar_preds = fig.colorbar(contour_preds, label=l)
cbar_labels.ax.set_ylabel(l, rotation=270, labelpad=15)
cbar_preds.ax.set_ylabel(l, rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

fig.savefig(f"{p['IMAGES_FOLDER']}/plot_comparison_{plot}_{date}.png")