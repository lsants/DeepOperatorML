import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import Normalize

f_index = 6

date = datetime.today().strftime('%Y%m%d')

with open('data_generation_params.yaml') as file:
    p_labels = yaml.safe_load(file)

with open('params_model.yaml') as file:
    p_preds = yaml.safe_load(file)

with open('data_generation_params.yaml') as file:
    p_labels = yaml.safe_load(file)

# ------------ Material constants ------------
Es = eval(p_labels['E'])
vs = p_labels['poisson']
e1 = Es/(1+vs)/(1-2*vs)
c44 = e1*(1-2*vs)/2
loadmag = p_labels['load']
load_stress = loadmag/(np.pi*p_labels['r_source']**2)

# ----------------Get Labels -----------------
# path = '/users/lsantia9/Documents/base script 160321/influencia_data.mat'
# data_labels = sio.loadmat(path)

data_labels = np.load(p_labels["SAVED_DATA_PATH"], allow_pickle=True)

# Extract variables and remove singleton dimensions
omega = np.squeeze(data_labels['freqs'])        # Shape: (num_freqs,)
r = np.squeeze(data_labels['r_field'])          # Shape: (n,)
z = np.squeeze(data_labels['z_field'])          # Shape: (n,)
wd = data_labels['wd']                           # Shape: (n, n, num_freqs)

# Print shapes for debugging
print('omega shape:', omega.shape)
print('r shape:', r.shape)
print('z shape:', z.shape)

wd = wd.reshape(len(omega), len(r), len(z))*(c44/(p_labels['r_source']*loadmag))
print('u shape:', wd.shape)

f_label_index = int(p_preds['TRAIN_PERC']*len(omega)) + f_index
print('label index is', f_label_index)

print

f_label = (p_labels['r_source']*(omega[f_label_index])*np.sqrt(p_labels['dens']/c44))
wd = wd[f_label_index, :, :]

wd_flip = np.flip(wd[1:, :], axis=0)
wd_full = np.concatenate((wd_flip, wd), axis=0)

wd_plot_abs = np.abs(wd_full)
l_abs = r'|$u_z$|'

wd_plot_real = np.real(wd_full)

l_real = r'Re($u_z$)'

wd_plot_imag = np.imag(wd_full)
l_imag = r'Im($u_z$)'

# -------------- Get Preds -------------------
data_preds = np.load(p_preds["TEST_PREDS_DATA_FILE"], allow_pickle=True)

u, xt, g_u_real, g_u_imag, mu_u, sd_u, mu_xt, sd_xt = data_preds["u"], data_preds["xt"], data_preds["real"], data_preds["imag"], data_preds['mu_u'], data_preds['sd_u'], data_preds['mu_xt'], data_preds['sd_xt']

f_pred_index = f_index
print('pred index is', f_pred_index)


f_pred = (p_labels['r_source']*(u.flatten()*sd_u + mu_u)*np.sqrt(p_labels['dens']/c44))[f_pred_index].item()

r_len = len(r)

r, z = (xt[:,0][:r_len]*sd_xt[0] + mu_xt[0])/p_labels['r_source'], (np.unique(xt[:,1])*sd_xt[1] + mu_xt[1])/p_labels['r_source']

r_full = np.concatenate((-np.flip(r[1:]), r))

g_u_real_normalized = (g_u_real.reshape(-1, len(r), len(z)))*(c44/(p_labels['r_source']*loadmag))
g_u_imag_normalized = (g_u_imag.reshape(-1, len(r), len(z)))*(c44/(p_labels['r_source']*loadmag))

print(g_u_real_normalized.shape)
print(f"Normalization parameters for branch: mu={mu_u}, sd={sd_u}")
print(f"Pred. freqs shape: {u.shape}")
print(f"Pred. coords shape: {xt.shape}")
print(f"Pred. g_u real shape: {g_u_real_normalized.shape}")
print(f"Pred. g_u imag shape: {g_u_imag_normalized.shape}")
print('\n')

g_u_abs = np.sqrt(g_u_real_normalized**2 + g_u_imag_normalized**2)

g_u_flip_abs = np.flip(g_u_abs[:, 1:, :], axis=0)
g_u_full_abs = np.concatenate((g_u_flip_abs, g_u_abs), axis=1)
g_u_full_abs_plot = g_u_full_abs[f_pred_index, :, :]

g_u_flip_real = np.flip(g_u_real_normalized[:, 1:, :], axis=1)
g_u_full_real = np.concatenate((g_u_flip_real, g_u_real_normalized), axis=1)
g_u_full_real_plot = g_u_full_real[f_pred_index, :, :]

g_u_flip_imag = np.flip(g_u_imag_normalized[:, 1:, :], axis=1)
g_u_full_imag = np.concatenate((g_u_flip_imag, g_u_imag_normalized), axis=1)
g_u_full_imag_plot = g_u_full_imag[f_pred_index, :, :]


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
print(f"Predicted/label frequencies: {f_pred:.1e}, {f_label:.1e}")
print(u*sd_u + mu_u, omega)
# logs = [eval(str(i)[:4]) for i in u.flatten()*sd_u + mu_u]
# print("Frequencies: ")
# print(*logs)
fig, ax = plt.subplots(nrows=3,
                       ncols=2,
                       figsize=(14, 10),
                       sharex='row',
                       sharey='row')

contour_preds_abs = ax[0][0].contourf(r_full,z,g_u_full_abs_plot.T,  cmap="viridis", norm=norm_abs)
ax[0][0].invert_yaxis()
ax[0][0].set_xlabel(r'$\frac{r}{a}$')
ax[0][0].set_ylabel(r'$\frac{z}{a}$')
ax[0][0].set_title(r'$|u_{z}|_{\mathrm{Pred}}$' + r' at $a_{0}$' + f'= {f_pred:.1e}')

contour_labels_abs = ax[0][1].contourf(r_full,z,wd_plot_abs.T,  cmap="viridis", norm=norm_abs)
ax[0][1].set_title(r'$|u_{z}|_{\mathrm{Label}}$' + r' at $a_{0}$' + f'= {f_label:.1e}')

contour_preds_real = ax[1][0].contourf(r_full,z,g_u_full_real_plot.T,  cmap="viridis", norm=norm_real)
ax[1][0].invert_yaxis()
ax[1][0].set_xlabel(r'$\frac{r}{a}$')
ax[1][0].set_ylabel(r'$\frac{z}{a}$')
ax[1][0].set_title(r'$\Re(u_{z})_{\mathrm{Pred}}$' + r' at $a_{0}$' + f'= {f_pred:.1e}')

contour_labels_real = ax[1][1].contourf(r_full,z,wd_plot_real.T,  cmap="viridis", norm=norm_real)
ax[1][1].set_title(r'$\Re(u_{z})_{\mathrm{Label}}$' + r' at $a_{0}$' + f'= {f_label:.1e}')

contour_preds_imag = ax[2][0].contourf(r_full,z,g_u_full_imag_plot.T,  cmap="viridis", norm=norm_imag)
ax[2][0].invert_yaxis()
ax[2][0].set_xlabel(r'$\frac{r}{a}$')
ax[2][0].set_ylabel(r'$\frac{z}{a}$')
ax[2][0].set_title(r'$\Im(u_{z})_{\mathrm{Pred}}$' + r' at $a_{0}$' + f'= {f_pred:.1e}')

contour_labels_imag = ax[2][1].contourf(r_full,z,wd_plot_imag.T,  cmap="viridis", norm=norm_imag)
ax[2][1].set_title(r'$\Im(u_{z})_{\mathrm{Label}}$' + r' at $a_{0}$' + f'= {f_label:.1e}')

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

fig.savefig(f"{p_preds['IMAGES_FOLDER']}/plot_comparison_{date}.png")