import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules.plotting import plot_field_comparison, plot_axis_comparison

f_index = 0

date = datetime.today().strftime('%Y%m%d')

with open('params_model.yaml') as file:
    p_preds = yaml.safe_load(file)

with open('data_generation_params.yaml') as file:
    p_labels = yaml.safe_load(file)

preds_datafile = p_preds["PREDS_DATA_FILE"]

if p_preds['DEBUG']:
    preds_datafile = preds_datafile[:-4] + "_" + date + '.npz'

# ----------------Get Labels -----------------
data_labels = np.load(p_labels["DATA_FILENAME"], allow_pickle=True)
print(f"Plotting labels from {p_labels['DATA_FILENAME']}")

omega = data_labels['xb']                # Shape: (num_freqs,)
r_label = data_labels['r']            # Shape: (n,)
z_label = data_labels['z']            # Shape: (m,)
wd = data_labels['g_u']                      # Shape: (num_freqs, n, m)

f_label_index = int(p_preds['TRAIN_PERC']*len(omega)) + f_index
f_label = omega[f_label_index]

print('omega shape:', omega.shape)
print('r labels shape:', r_label.shape)
print('z labels shape:', z_label.shape)
print('u shape:', wd.shape)
print("-----------------------")

# -------------- Get Preds -------------------
data_preds = np.load(preds_datafile, allow_pickle=True)
print(f"Plotting predictions from {preds_datafile}")

u = data_preds["u"].flatten()                                # Shape: (test_size, 1)
xt = data_preds["xt"]                              # Shape: (n*m, 2)
g_u_real = data_preds["real"]                      # Shape: (test_size, n*m)
g_u_imag = data_preds["imag"]                      # Shape: (test_size, n*m)
mu_u = data_preds['mu_u']                          # Shape: (1,)
sd_u = data_preds['sd_u']                          # Shape: (1,)
mu_xt = data_preds['mu_xt']                        # Shape: (2,)
sd_xt  = data_preds['sd_xt']                       # Shape: (2,)

r_pred, z_pred = (xt[:,0][:p_labels['N_R']]), (np.unique(xt[:,1]))

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
assert f_pred - f_label < 1e-14, "Frequencies to plot are different between label and prediction"

r, z = r_pred, z_pred = r_label, z_label

freq = f_pred = f_label

wd_plot = wd[f_label_index]
g_u_plot = g_u[f_pred_index]


print(f"Shapes of plotted variables:\n Labels={wd_plot.shape}\n Preds={g_u_plot.shape}")

fig = plot_field_comparison(r, z, wd_plot, g_u_plot, freq)
plt.show()

fig.savefig(f"{p_preds['IMAGES_FOLDER']}/plot_field_comparison_{date}.png")

fig = plot_axis_comparison(r, z, wd_plot, g_u_plot, freq)
plt.show()

fig.savefig(f"{p_preds['IMAGES_FOLDER']}/plot_axis_comparison_{date}.png")