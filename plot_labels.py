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

# ----------------Get Labels -----------------
data_labels = np.load(p_labels['SAVED_DATA_PATH'], allow_pickle=True)

omega = data_labels['freqs']                # Shape: (num_freqs,)
r_label = data_labels['r_field']            # Shape: (n,)
z_label = data_labels['z_field']            # Shape: (m,)
wd = data_labels['wd']                      # Shape: (num_freqs, n, m)

print('omega shape:', omega.shape)
print('r labels shape:', r_label.shape)
print('z labels shape:', z_label.shape)
print('u shape:', wd.shape)

f_label_index = int(p_preds['TRAIN_PERC']*len(omega)) + f_index
f_label = omega[f_label_index]

r, z = r_label, z_label

if p_labels['non_dim']:
    r, z = r * p_labels['r_source'], z * p_labels['r_source']

freq = f_label

wd_plot = wd[f_label_index]


fig = plot_label_contours(r, z, wd_plot, freq, full=True, non_dim_plot=p_labels['non_dim'])
plt.show()

fig = plot_label_axis(r, z, wd_plot, freq, axis=p_labels['axis_plot'], non_dim_plot=p_labels['non_dim'])

plt.show()