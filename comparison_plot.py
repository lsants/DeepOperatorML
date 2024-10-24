import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules.plotting import plot_comparison

f_index = 0

date = datetime.today().strftime('%Y%m%d')

with open('params_model.yaml') as file:
    p_preds = yaml.safe_load(file)

with open('data_generation_params.yaml') as file:
    p_labels = yaml.safe_load(file)

normalized = p_labels['non_dim']

preds_datafile = p_preds["PREDS_DATA_FILE"]

if p_preds['DEBUG']:
    preds_datafile = preds_datafile[:-4] + "_" + date + '.npz'

# ----------------Get Labels -----------------
data_labels = np.load(p_labels["SAVED_DATA_PATH"], allow_pickle=True)

omega = data_labels['freqs']                # Shape: (num_freqs,)
r_label = data_labels['r_field']            # Shape: (n,)
z_label = data_labels['z_field']            # Shape: (m,)
wd = data_labels['wd']                      # Shape: (num_freqs, n, m)

f_label_index = int(p_preds['TRAIN_PERC']*len(omega)) + f_index
f_label = omega[f_label_index]

print('omega shape:', omega.shape)
print('r labels shape:', r_label.shape)
print('z labels shape:', z_label.shape)
print('u shape:', wd.shape)
print("-----------------------")

# -------------- Get Preds -------------------
data_preds = np.load(preds_datafile, allow_pickle=True)

u = data_preds["u"].flatten()                                # Shape: (test_size, 1)
xt = data_preds["xt"]                              # Shape: (n*m, 2)
g_u_real = data_preds["real"]                      # Shape: (test_size, n*m)
g_u_imag = data_preds["imag"]                      # Shape: (test_size, n*m)
mu_u = data_preds['mu_u']                          # Shape: (1,)
sd_u = data_preds['sd_u']                          # Shape: (1,)
mu_xt = data_preds['mu_xt']                        # Shape: (2,)
sd_xt  = data_preds['sd_xt']                       # Shape: (2,)

r_pred, z_pred = (xt[:,0][:p_labels['n_r']]*sd_xt[0] + mu_xt[0]), (np.unique(xt[:,1])*sd_xt[1] + mu_xt[1])

f_pred_index = f_index
f_pred = (u*sd_u + mu_u)[f_pred_index].item()

g_u = g_u_real + g_u_imag*1j
g_u = g_u.reshape(-1, len(r_pred), len(z_pred))

print(f"Pred. freqs shape: {u.shape}")
print(f"Pred. coords shape: {xt.shape}")
print('r preds shape:', r_pred.shape)
print('z preds shape:', z_pred.shape)
print(f'shapes: {g_u_real.shape}, {g_u_imag.shape}')

print('\n')

# --------- Check domains --------------
assert ((abs(r_pred - r_label) < 1e-14).all() and (abs(z_pred - z_label < 1e-14)).all()), "Domains are different for labels and predictions"
assert f_pred == f_label, "Frequencies to plot are different between label and prediction"

r, z = r_pred, z_pred = r_label, z_label
freq = f_pred = f_label

wd_plot = wd[f_label_index]
g_u_plot = g_u[f_pred_index]

fig = plot_comparison(r, z, wd_plot, g_u_plot, freq, full=True, non_dim_plot=normalized)

plt.tight_layout()
plt.show()

fig.savefig(f"{p_preds['IMAGES_FOLDER']}/plot_comparison_{date}.png")