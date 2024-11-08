import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules import dir_functions
from modules import preprocessing as ppr
from modules.greenfunc_dataset import GreenFuncDataset
from modules.plotting import plot_axis, plot_field, plot_labels_axis

with open('params_test.yaml') as file:
    p = yaml.safe_load(file)

# ----------------Get Labels -----------------
data = np.load(p['DATAFILE'], allow_pickle=True)

print(f"Plotting data from: {p['DATAFILE']}")

dataset = GreenFuncDataset(data)
indices = dir_functions.load_data_info(p['INDICES_FILE'])
test_indices = indices['test']
test_dataset = dataset[test_indices]

xb = test_dataset['xb']
xt = dataset.get_trunk()
r, z = ppr.trunk_to_meshgrid(xt)

g_u_real = test_dataset['g_u_real']
g_u_imag = test_dataset['g_u_imag']
g_u = g_u_real + g_u_imag * 1j
g_u = ppr.reshape_from_model(g_u, z)

quantiles = np.quantile(xb, np.arange(0.1, 1.1, 0.1)) # Plot each 10th quantile frequency

mask = [(np.abs(xb - i)).argmin() for i in quantiles]

fig_labels = plot_labels_axis(r, z, g_u[mask], xb[mask], non_dim_plot=True)
plt.show()

# Add animations for field and axis plots