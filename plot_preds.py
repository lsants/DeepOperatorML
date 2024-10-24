import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules.plotting import plot_label_axis, plot_label_contours

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

# ----------------Get Preds -----------------
data_preds = np.load(preds_datafile, allow_pickle=True)

u = data_preds["u"].flatten()                                # Shape: (test_size, 1)
xt = data_preds["xt"]                              # Shape: (n*m, 2)
g_u_real = data_preds["real"]                      # Shape: (test_size, n*m)
g_u_imag = data_preds["imag"]                      # Shape: (test_size, n*m)
mu_u = data_preds['mu_u']                          # Shape: (1,)
sd_u = data_preds['sd_u']                          # Shape: (1,)
mu_xt = data_preds['mu_xt']                        # Shape: (2,)
sd_xt  = data_preds['sd_xt']                       # Shape: (2,)

r_pred, z_pred = (xt[:,0][:p_labels['n_r']]), (np.unique(xt[:,1]))

f_pred_index = f_index

real_u = u*sd_u + mu_u
f_pred = real_u[f_pred_index].item()

g_u = g_u_real + g_u_imag*1j
g_u = g_u.reshape(-1, len(r_pred), len(z_pred))

r, z = r_pred, z_pred

if p_labels['non_dim']:
    r, z = r_pred / p_labels['r_source'], z_pred / p_labels['r_source']

freq = f_pred
g_u_plot = g_u[f_pred_index]

fig = plot_label_contours(r, z, g_u_plot, freq, full=True, non_dim_plot=p_labels['non_dim'])
plt.show()

fig = plot_label_axis(r, z, g_u_plot, freq, axis=p_labels['axis_plot'], non_dim_plot=p_labels['non_dim'])

plt.show()