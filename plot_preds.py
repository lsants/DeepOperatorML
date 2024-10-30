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

preds_datafile = p_preds["PREDS_DATA_FILE"]

if p_preds['DEBUG']:
    preds_datafile = preds_datafile[:-4] + "_" + date + '.npz'

# ----------------Get Preds -----------------
data_preds = np.load(preds_datafile, allow_pickle=True)

print(f"Plotting from: {preds_datafile}")

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

real_u = u*sd_u + mu_u
f_pred = real_u[f_pred_index].item()

g_u = g_u_real + g_u_imag*1j
g_u = g_u.reshape(-1, len(r_pred), len(z_pred))

r, z = r_pred, z_pred

freq = f_pred
g_u_plot = g_u[f_pred_index]

fig = plot_label_contours(r, z, g_u_plot, freq, full=True)
plt.show()

fig = plot_label_axis(r, z, g_u_plot, freq)

plt.show()