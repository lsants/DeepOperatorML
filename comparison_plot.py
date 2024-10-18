import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from datetime import datetime
from matplotlib.colors import Normalize

f_index = 0

with open('params_model.yaml') as file:
    p = yaml.safe_load(file)

date = datetime.today().strftime('%Y%m%d')

# ----------------Get Labels -----------------
# path = '/users/lsantia9/Documents/base script 160321/influencia_data.mat'
# data_labels = sio.loadmat(path)

data_labels = np.load("./data/raw/data_damped.npz", allow_pickle=True)

# Extract variables and remove singleton dimensions
omega = np.squeeze(data_labels['freqs'])        # Shape: (num_freqs,)
r = np.squeeze(data_labels['r_field'])          # Shape: (n,)
z = np.squeeze(data_labels['z_field'])          # Shape: (n,)
wd = data_labels['wd'].transpose(0,1,2)                           # Shape: (n, n, num_freqs)

# Print shapes for debugging
print('omega shape:', omega.shape)
print('r shape:', r.shape)
print('z shape:', z.shape)
print('u shape:', wd.shape)

wd = wd.reshape(len(r), len(z), len(omega))

f_label_index = len(omega) - int(p['TRAIN_PERC']*len(omega)) + f_index
f_label = omega[f_label_index]
wd = wd[:,:,f_label_index]

wd_flip = np.flip(wd[1:, :], axis=0)
wd_full = np.concatenate((wd_flip, wd), axis=0)

wd_plot_abs = np.abs(wd_full)
l_abs = r'|$u_z$|'

wd_plot_real = np.real(wd_full)

l_real = r'Re($u_z$)'

wd_plot_imag = np.imag(wd_full)
l_imag = r'Im($u_z$)'

# -------------- Get Preds -------------------
data_preds = np.load("./data/output/test_output.npz", allow_pickle=True)

u, xt, g_u_real, g_u_imag, mu_u, sd_u, mu_xt, sd_xt = data_preds["u"], data_preds["xt"], data_preds["real"], data_preds["imag"], data_preds['mu_u'], data_preds['sd_u'], data_preds['mu_xt'], data_preds['sd_xt']

f_pred_index = f_index
f_pred = (u.flatten()*sd_u + mu_u)[f_pred_index].item()

r_len = len(r)

r, z = xt[:,0][:r_len]*sd_xt[0] + mu_xt[0], np.unique(xt[:,1])*sd_xt[1] + mu_xt[1]

r_full = np.concatenate((-np.flip(r[1:]), r))

g_u_real, g_u_imag = g_u_real.reshape(len(r), len(z), -1), g_u_imag.reshape(len(r), len(z), -1)

print(f"Normalization parameters for branch: mu={mu_u}, sd={sd_u}")
print(f"Pred. freqs shape: {u.shape}")
print(f"Pred. coords shape: {xt.shape}")
print(f"Pred. g_u real shape: {g_u_real.shape}")
print(f"Pred. g_u imag shape: {g_u_imag.shape}")
print('\n')

g_u_abs = np.sqrt(g_u_real**2 + g_u_imag**2)


g_u_flip_abs = np.flip(g_u_abs[1:, :, :], axis=0)
g_u_full_abs = np.concatenate((g_u_flip_abs, g_u_abs), axis=0)
g_u_full_abs_plot = g_u_full_abs[:, :, f_pred_index]

g_u_flip_real = np.flip(g_u_real[1:, :, :], axis=0)
g_u_full_real = np.concatenate((g_u_flip_real, g_u_real), axis=0)
g_u_full_real_plot = g_u_full_real[:, :, f_pred_index]

g_u_flip_imag = np.flip(g_u_imag[1:, :, :], axis=0)
g_u_full_imag = np.concatenate((g_u_flip_imag, g_u_imag), axis=0)
g_u_full_imag_plot = g_u_full_imag[:, :, f_pred_index]


print('\n')


#  --------------- Plots ----------------
abs_min = min(np.min(g_u_full_abs_plot), np.min(wd_plot_abs))
abs_max = max(np.max(g_u_full_abs_plot), np.max(wd_plot_abs))

real_min = min(np.min(g_u_full_real_plot), np.min(wd_plot_real))
real_max = max(np.max(g_u_full_real_plot), np.max(wd_plot_real))

imag_min = min(np.min(g_u_full_imag_plot), np.min(wd_plot_imag))
imag_max = max(np.max(g_u_full_imag_plot), np.max(wd_plot_imag))

norm_abs = Normalize(vmin=abs_min, vmax=abs_max)
norm_real = Normalize(vmin=real_min, vmax=real_max)
norm_imag = Normalize(vmin=imag_min, vmax=imag_max)

print("-------------------------")
# print(f"Predicted/label frequencies: {f_pred:.2f}, {f_label:.2f}")
logs = [eval(str(i)[:4]) for i in u.flatten()*sd_u + mu_u]
print("Frequencies: ")
print(*logs)
fig, ax = plt.subplots(nrows=3,
                       ncols=2,
                       figsize=(14, 10),
                       sharex='row',
                       sharey='row')

contour_preds_abs = ax[0][0].contourf(r_full,z,g_u_full_abs_plot.T,  cmap="viridis", norm=norm_abs)
ax[0][0].invert_yaxis()
ax[0][0].set_xlabel('r')
ax[0][0].set_ylabel('z')
ax[0][0].set_title(r'$|u_{z}|_{\mathrm{Pred}}$' + f' at ω = {f_pred:.2f} rad/s')

contour_labels_abs = ax[0][1].contourf(r_full,z,wd_plot_abs.T,  cmap="viridis", norm=norm_abs)
ax[0][1].set_title(r'$|u_{z}|_{\mathrm{Label}}$' + f' at ω = {f_label:.2f} rad/s')

contour_preds_real = ax[1][0].contourf(r_full,z,g_u_full_real_plot.T,  cmap="viridis", norm=norm_real)
ax[1][0].invert_yaxis()
ax[1][0].set_xlabel('r')
ax[1][0].set_ylabel('z')
ax[1][0].set_title(r'$\Re(u_{z})_{\mathrm{Pred}}$' + f' at ω = {f_pred:.2f} rad/s')

contour_labels_real = ax[1][1].contourf(r_full,z,wd_plot_real.T,  cmap="viridis", norm=norm_real)
ax[1][1].set_title(r'$\Re(u_{z})_{\mathrm{Label}}$' + f' at ω = {f_label:.2f} rad/s')

contour_preds_imag = ax[2][0].contourf(r_full,z,g_u_full_imag_plot.T,  cmap="viridis", norm=norm_imag)
ax[2][0].invert_yaxis()
ax[2][0].set_xlabel('r')
ax[2][0].set_ylabel('z')
ax[2][0].set_title(r'$\Im(u_{z})_{\mathrm{Pred}}$' + f' at ω = {f_pred:.2f} rad/s')

contour_labels_imag = ax[2][1].contourf(r_full,z,wd_plot_imag.T,  cmap="viridis", norm=norm_imag)
ax[2][1].set_title(r'$\Im(u_{z})_{\mathrm{Label}}$' + f' at ω = {f_label:.2f} rad/s')

cbar_labels_abs = fig.colorbar(contour_labels_abs, label=l_abs, ax=ax[0][1], norm=norm_abs)
cbar_preds_abs = fig.colorbar(contour_preds_abs, label=l_abs, ax=ax[0][0], norm=norm_abs)
cbar_labels_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)
cbar_preds_abs.ax.set_ylabel(l_abs, rotation=270, labelpad=15)

cbar_labels_real = fig.colorbar(contour_labels_real, label=l_real, ax=ax[1][1], norm=norm_real)
cbar_preds_real = fig.colorbar(contour_preds_real, label=l_real, ax=ax[1][0], norm=norm_real)
cbar_labels_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)
cbar_preds_real.ax.set_ylabel(l_real, rotation=270, labelpad=15)

cbar_labels_imag = fig.colorbar(contour_labels_imag, label=l_imag, ax=ax[2][1], norm=norm_imag)
cbar_preds_imag = fig.colorbar(contour_preds_imag, label=l_imag, ax=ax[2][0], norm=norm_imag)
cbar_labels_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)
cbar_preds_imag.ax.set_ylabel(l_imag, rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

fig.savefig(f"{p['IMAGES_FOLDER']}/plot_comparison_{date}.png")