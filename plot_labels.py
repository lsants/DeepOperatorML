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

# ----------------Get Labels -----------------
data_labels = np.load(p_labels['DATA_FILENAME'], allow_pickle=True)

print(f"Plotting data from: {p_labels['DATA_FILENAME']}")

omega = data_labels['xb']                # Shape: (num_freqs,)
r_label = data_labels['r']            # Shape: (n,)
z_label = data_labels['z']            # Shape: (m,)
wd = data_labels['g_u']                      # Shape: (num_freqs, n, m)

print('omega shape:', omega.shape)
print('r labels shape:', r_label.shape)
print('z labels shape:', z_label.shape)
print('u shape:', wd.shape)

f_label_index = int(p_preds['TRAIN_PERC']*len(omega)) + f_index
f_label = omega[f_label_index]

r, z = r_label, z_label

freq = f_label

wd_plot = wd[f_label_index]


fig = plot_label_contours(r, z, wd_plot, freq, full=True, non_dim_plot=True)
plt.show()

fig = plot_label_axis(r, z, wd_plot, freq, non_dim_plot=True)

plt.show()