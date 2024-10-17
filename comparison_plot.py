import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from datetime import datetime

with open('params_model.yaml') as file:
    p = yaml.safe_load(file)

date = datetime.today().strftime('%Y%m%d')

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

freq_index = -1
wd = wd[:,:,freq_index]

wd_flip = np.flip(wd[1:, :], axis=0)
wd_full = np.concatenate((wd_flip, wd), axis=0)

# -------------- Get Preds -------------------
data_preds = np.load("./data/output/test_output.npz", allow_pickle=True)

u, xt, g_u_real, g_u_imag, mu_u, sd_u, mu_xt, sd_xt = data_preds["u"], data_preds["xt"], data_preds["real"], data_preds["imag"], data_preds['mu_u'], data_preds['sd_u'], data_preds['mu_xt'], data_preds['sd_xt']

f_pred = (u[0].flatten()*sd_u + mu_u).item()

fix_this = 8

r, z = xt[:,0][:fix_this]*sd_xt[0] + mu_xt[0], np.unique(xt[:,1])*sd_xt[1] + mu_xt[1]

r_full = np.concatenate((-np.flip(r[1:]), r))

g_u_real, g_u_imag = g_u_real.reshape(len(r), len(z)), g_u_imag.reshape(len(r), len(z))

print(f"Test freqs: {u.flatten()*sd_u + mu_u}")
print(f"Test coords: {xt.shape}")
print(f"Test g_u real: {g_u_real.shape}")
print(f"Test g_u imag: {g_u_imag.shape}")

g_u_abs = np.sqrt(g_u_real**2 + g_u_imag**2)

wd_plot_abs = np.abs(wd_full)
l_abs = r'|$u_z$|'

wd_plot_real = np.real(wd_full)

l_real = r'Re($u_z$)'

wd_plot_imag = np.imag(wd_full)
l_imag = r'Im($u_z$)'

g_u_flip_abs = np.flip(g_u_abs[1:, :], axis=0)
g_u_full_abs = np.concatenate((g_u_flip_abs, g_u_abs), axis=0)

g_u_flip_real = np.flip(g_u_real[1:, :], axis=0)
g_u_full_real = np.concatenate((g_u_flip_real, g_u_real), axis=0)

g_u_flip_imag = np.flip(g_u_imag[1:, :], axis=0)
g_u_full_imag = np.concatenate((g_u_flip_imag, g_u_imag), axis=0)

#  --------------- Plots ----------------
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))

contour_preds_abs = ax[0][0].contourf(r_full,z,g_u_full_abs.T,  cmap="viridis")
ax[0][0].invert_yaxis()
ax[0][0].set_xlabel('r')
ax[0][0].set_ylabel('z')
ax[0][0].set_title(r'$|u_{z}|_{\mathrm{Pred}}$' + f' at ω = {f_pred:.2f} rad/s')

contour_labels_abs = ax[0][1].contourf(r_full,z,wd_plot_abs.T,  cmap="viridis")
ax[0][1].invert_yaxis()
ax[0][1].set_xlabel('r')
ax[0][1].set_ylabel('z')
ax[0][1].set_title(r'$|u_{z}|_{\mathrm{Label}}$' + f' at ω = {omega[freq_index]:.2f} rad/s')

contour_preds_real = ax[1][0].contourf(r_full,z,g_u_full_real.T,  cmap="viridis")
ax[1][0].invert_yaxis()
ax[1][0].set_xlabel('r')
ax[1][0].set_ylabel('z')
ax[1][0].set_title(r'$\Re(u_{z})_{\mathrm{Pred}}$' + f' at ω = {f_pred:.2f} rad/s')

contour_labels_real = ax[1][1].contourf(r_full,z,wd_plot_real.T,  cmap="viridis")
ax[1][1].invert_yaxis()
ax[1][1].set_xlabel('r')
ax[1][1].set_ylabel('z')
ax[1][1].set_title(r'$\Re(u_{z})_{\mathrm{Label}}$' + f' at ω = {omega[freq_index]:.2f} rad/s')

contour_preds_imag = ax[2][0].contourf(r_full,z,g_u_full_imag.T,  cmap="viridis")
ax[2][0].invert_yaxis()
ax[2][0].set_xlabel('r')
ax[2][0].set_ylabel('z')
ax[2][0].set_title(r'$\Im(u_{z})_{\mathrm{Pred}}$' + f' at ω = {f_pred:.2f} rad/s')

contour_labels_imag = ax[2][1].contourf(r_full,z,wd_plot_imag.T,  cmap="viridis")
ax[2][1].invert_yaxis()
ax[2][1].set_xlabel('r')
ax[2][1].set_ylabel('z')
ax[2][1].set_title(r'$\Im(u_{z})_{\mathrm{Label}}$' + f' at ω = {omega[freq_index]:.2f} rad/s')


cbar_labels_abs = fig.colorbar(contour_labels_abs, label=l_abs)
cbar_preds_abs = fig.colorbar(contour_preds_abs, label=l_abs)
cbar_labels_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)
cbar_preds_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)

cbar_labels_real = fig.colorbar(contour_labels_real, label=l_real)
cbar_preds_real = fig.colorbar(contour_preds_real, label=l_real)
cbar_labels_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)
cbar_preds_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)

cbar_labels_imag = fig.colorbar(contour_labels_imag, label=l_imag)
cbar_preds_imag = fig.colorbar(contour_preds_imag, label=l_imag)
cbar_labels_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)
cbar_preds_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

fig.savefig(f"{p['IMAGES_FOLDER']}/plot_comparison_{date}.png")