import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules.plotting import plot_axis, plot_field

f_index = 4

date = datetime.today().strftime('%Y%m%d')

with open('data_generation_params.yaml') as file:
    p_labels = yaml.safe_load(file)

# ----------------Get Labels -----------------
data_labels = np.load(p_labels['DATA_FILENAME'], allow_pickle=True)

print(f"Plotting data from: {p_labels['DATA_FILENAME']}")

omega = data_labels['xb']             # Shape: (num_freqs,)
r_label = data_labels['r']            # Shape: (n,)
z_label = data_labels['z']            # Shape: (m,)
wd = data_labels['g_u']               # Shape: (num_freqs, n, m)

print('omega shape:', omega.shape)
print('r labels shape:', r_label.shape)
print('z labels shape:', z_label.shape)
print('u shape:', wd.shape)

f_label = omega[f_index]

r, z = r_label, z_label

freq = f_label

wd_plot = wd[f_index]

fig = plot_field(r, z, wd_plot, freq)
plt.show()

fig = plot_axis(r, z, wd_plot, freq)

plt.show()